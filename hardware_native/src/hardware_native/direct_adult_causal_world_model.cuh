#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CAUSAL_WORLD_MODEL_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CAUSAL_WORLD_MODEL_CUH

// h.causal_world_model (#1611). This is a bounded resident projection/receipt
// of causal relations realized by the Adult's mixed-rank relational Network,
// not a parallel knowledge ontology. Relations between what this subject did
// and what the world returned are grown from settled verified consequences:
// every edge keys on the emitted action and ancestry-resolved sensory root,
// and only complete-ancestry interventions may mint one. Prediction is a
// majority vote over lived outcomes that fails closed for unknown pairs.

#include <cstdint>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kCausalModelRelationCapacity = 16u;
inline constexpr std::uint32_t kCausalModelOutcomeBuckets = 4u;
inline constexpr std::uint32_t kCausalModelMinimumSupport = 2u;
// Journal depth above which per-pass world-return compaction beats the
// per-ticket full scans; below it the original resolver stays cheaper than
// building the index. Measured crossover on the host replica sits between
// 128 and 512 records.
inline constexpr std::uint32_t kCausalModelReturnIndexMinimumRecords = 256u;

struct DirectCausalRelation {
  std::uint32_t action_value;
  std::uint32_t root_channel;
  std::uint32_t outcome_values[kCausalModelOutcomeBuckets] = {};
  std::uint32_t outcome_counts[kCausalModelOutcomeBuckets] = {};
  std::uint32_t observations;
  // Current-world reread state: the newest verified return for this relation
  // and the tick that carried it. Distinct from the majority prediction --
  // this is what later action and report must READ, not assume.
  std::uint32_t current_outcome_value;
  std::uint32_t current_outcome_tick;
};

struct DirectCausalWorldModel {
  DirectCausalRelation relations[kCausalModelRelationCapacity] = {};
  std::uint32_t relation_count;
  std::uint32_t authority_refusals;
  std::uint32_t revisions;
  std::uint64_t processed_ticket_by_slot[direct_adult_core::kMaxAsynchronousTickets] = {};
  std::uint64_t model_identity;
};

static_assert(std::is_trivially_copyable_v<DirectCausalWorldModel>);

__host__ __device__ inline std::uint64_t causal_model_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

__host__ __device__ inline std::int32_t causal_model_find_relation(
    const DirectCausalWorldModel& model, std::uint32_t action_value,
    std::uint32_t root_channel) {
  for (std::uint32_t i = 0u; i < model.relation_count; ++i)
    if (model.relations[i].action_value == action_value &&
        model.relations[i].root_channel == root_channel)
      return static_cast<std::int32_t>(i);
  return -1;
}

// World-return compaction. Both derivers resolve one verified world return
// per unsettled ticket by scanning the whole journal per ticket -- quadratic
// in history depth for a single-thread device kernel. The journal prefix is
// immutable across the pass and the resolver is pure, so the world_return
// records are collected once and every per-ticket resolution then visits the
// same records in the same order with the same predicate, minus only the
// records the original scan provably skips (kind != world_return).
struct DirectCausalWorldReturnIndex {
  std::uint32_t count;
  std::uint32_t record_index[kDirectExactHistoryHotPageCapacity];
};

// Returns false when the journal cannot be compacted (oversized count); the
// caller then falls back to the original full-scan resolver.
__device__ inline bool causal_model_build_return_index(
    const DirectExactHistoryRecord* records, std::uint32_t record_count,
    DirectCausalWorldReturnIndex* index) {
  index->count = 0u;
  if (record_count > kDirectExactHistoryHotPageCapacity) return false;
  for (std::uint32_t i = 0u; i < record_count; ++i)
    if (records[i].kind == DirectExactHistoryKind::world_return)
      index->record_index[index->count++] = i;
  return true;
}

// Bit-equivalent re-expression of affect_verified_world_return
// (direct_adult_affect_body.cuh) over that compaction: identical visitation
// order, match counting, early refusal on any non-exact candidate, and
// returned pointer. The exactness predicate below must stay textually
// aligned with that resolver; it is its semantic twin, not a replacement.
__device__ inline const DirectExactHistoryRecord*
causal_model_verified_world_return(
    const direct_adult_core::AsynchronousTicket& ticket,
    const DirectExactHistoryRecord* records,
    const DirectCausalWorldReturnIndex& index) {
  const DirectExactHistoryRecord* found = nullptr;
  std::uint32_t matches = 0u;
  for (std::uint32_t k = 0u; k < index.count; ++k) {
    const DirectExactHistoryRecord& record = records[index.record_index[k]];
    if (record.identity != ticket.ticket_id) continue;
    ++matches;
    const bool exact =
        record.parent_identity == ticket.upstream_ticket_id &&
        record.source == ticket.motor_node &&
        record.subject == ticket.motor_channel &&
        record.context == ticket.context_signature &&
        record.event_tick == record.resident_tick &&
        record.resident_tick >= ticket.emission_tick &&
        (record.flags & kDirectHistoryVerifiedObservation) != 0u &&
        (record.flags & kDirectHistoryPayloadFlags) ==
            (ticket.mismatch_bits & kDirectHistoryPayloadFlags) &&
        record.resource_delta == ticket.settled_reward_q16;
    if (!exact) return nullptr;
    found = &record;
  }
  return matches == 1u ? found : nullptr;
}

// Derive relations from the settlement ledgers. One pass over ticket slots,
// each depositing at most once per ticket identity while lawful slot reuse by
// a new ticket stays legal. An intervention needs an exact verified world
// return AND a complete participation ancestry; anything else is counted as
// an authority refusal and mints nothing.
__device__ inline void derive_causal_world_relations(
    DirectCausalWorldModel* model,
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, const DirectExactHistoryRecord* records,
    std::uint32_t record_count) {
  if (model == nullptr || tickets == nullptr || records == nullptr) return;
  const std::uint32_t bounded =
      ticket_count < direct_adult_core::kMaxAsynchronousTickets
          ? ticket_count
          : direct_adult_core::kMaxAsynchronousTickets;
  DirectCausalWorldReturnIndex return_index;  // only [0, count) is read
  bool return_index_ready = false, return_index_usable = false;
  for (std::uint32_t slot = 0u; slot < bounded; ++slot) {
    const direct_adult_core::AsynchronousTicket& ticket = tickets[slot];
    if (ticket.ticket_id == 0u || ticket.settled == 0u ||
        model->processed_ticket_by_slot[slot] == ticket.ticket_id)
      continue;
    if (!return_index_ready) {
      return_index_ready = true;
      return_index_usable =
          record_count >= kCausalModelReturnIndexMinimumRecords &&
          causal_model_build_return_index(records, record_count, &return_index);
    }
    const DirectExactHistoryRecord* consequence =
        return_index_usable
            ? causal_model_verified_world_return(ticket, records, return_index)
            : affect_verified_world_return(ticket, records, record_count);
    if (consequence == nullptr || ticket.upstream_ticket_id ==
                                              direct_adult_core::kInvalidTicket) {
      ++model->authority_refusals;
      continue;
    }
    std::uint32_t root_channel = 0u;
    if (!affect_root_contact_channel(records, record_count, ticket.ticket_id,
                                     &root_channel)) {
      ++model->authority_refusals;
      continue;
    }
    std::int32_t index =
        causal_model_find_relation(*model, ticket.motor_word, root_channel);
    if (index < 0) {
      if (model->relation_count >= kCausalModelRelationCapacity) {
        ++model->authority_refusals;
        continue;
      }
      index = static_cast<std::int32_t>(model->relation_count);
      DirectCausalRelation fresh{};
      fresh.action_value = ticket.motor_word;
      fresh.root_channel = root_channel;
      model->relations[index] = fresh;
      ++model->relation_count;
    }
    DirectCausalRelation& relation = model->relations[index];
    // Standing majority before this observation lands: an observation
    // against it is a counted revision of what the model expected.
    std::uint32_t standing_best = kCausalModelOutcomeBuckets;
    std::uint32_t standing_best_count = 0u;
    for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b)
      if (relation.outcome_counts[b] > standing_best_count) {
        standing_best_count = relation.outcome_counts[b];
        standing_best = b;
      }
    bool known_bucket = false;
    for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b) {
      if (relation.outcome_counts[b] != 0u &&
          relation.outcome_values[b] == consequence->value) {
        if (standing_best_count > 0u && b != standing_best)
          ++model->revisions;
        ++relation.outcome_counts[b];
        known_bucket = true;
        break;
      }
    }
    if (!known_bucket) {
      std::uint32_t empty_slot = kCausalModelOutcomeBuckets;
      for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b)
        if (relation.outcome_counts[b] == 0u) {
          empty_slot = b;
          break;
        }
      if (empty_slot == kCausalModelOutcomeBuckets) {
        ++model->authority_refusals;
        continue;
      }
      relation.outcome_values[empty_slot] = consequence->value;
      relation.outcome_counts[empty_slot] = 1u;
      if (relation.observations > 0u) ++model->revisions;
    }
    ++relation.observations;
    if (consequence->resident_tick >= relation.current_outcome_tick) {
      relation.current_outcome_value =
          static_cast<std::uint32_t>(consequence->value);
      relation.current_outcome_tick = consequence->resident_tick;
    }
    model->processed_ticket_by_slot[slot] = ticket.ticket_id;
    model->model_identity = causal_model_fold(
        causal_model_fold(model->model_identity, ticket.ticket_id),
        (static_cast<std::uint64_t>(ticket.motor_word) << 32u) |
            root_channel);
  }
}


// Action-participation twin of the deriver above. The emitted action freezes
// the exact participants that caused it; those links are durable resident
// evidence and survive until settlement/checkpoint, unlike transient executor
// staging. Resolve sensory roots only when each frozen participant still joins
// to its resident contact authority.
__device__ inline bool causal_model_resident_participant_root_channel(
    const ResidentPostbirthConstructorState* constructor_state,
    const direct_adult_core::DirectActionParticipationLink& link,
    std::uint32_t action_emission_tick, std::uint32_t* root_channel) {
  using direct_adult_core::DirectContributionKind;
  const bool sparse =
      link.contribution_kind == DirectContributionKind::sparse_route;
  if (constructor_state == nullptr || root_channel == nullptr ||
      link.participant_ticket_id == 0u ||
      link.participant_ticket_id == direct_adult_core::kInvalidTicket ||
      link.authority !=
          direct_adult_core::DirectParticipationAuthority::independent_external ||
      (link.contribution_kind != DirectContributionKind::direct_ingress &&
       !sparse) ||
      link.expiry_tick < action_emission_tick || link.claim_incarnation == 0u ||
      link.authority_incarnation == 0u ||
      (sparse &&
       (link.route_index == direct_adult_core::kInvalidIndex ||
        link.route_incarnation == 0u || link.frozen_eligibility_q16 <= 0 ||
        link.eligibility_slot == direct_adult_core::kInvalidIndex ||
        link.eligibility_generation == 0u)) ||
      constructor_state->raw_contact_binding_count > kResidentRecipeIncidenceCapacity)
    return false;
  const ResidentRawContactBinding binding = resident_raw_contact_authority(
      *constructor_state, link.participant_ticket_id);
  if (binding.ticket_id != link.participant_ticket_id ||
      resident_raw_contact_key_empty(binding.key) ||
      binding.authority.claim_incarnation != link.claim_incarnation ||
      binding.authority.authority_incarnation != link.authority_incarnation ||
      !resident_sensory_authority_receipt_valid(
          binding.ticket_id, binding.key, binding.authority,
          static_cast<std::uint32_t>(
              direct_adult_core::CausalOrigin::external_contact)))
    return false;
  *root_channel = binding.key.channel;
  return true;
}

__device__ inline void derive_causal_world_relations_from_action_participation(
    DirectCausalWorldModel* model,
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, const DirectExactHistoryRecord* records,
    std::uint32_t record_count, const ResidentPostbirthConstructorState* constructor_state,
    const direct_adult_core::DirectActionOccurrence* actions,
    const direct_adult_core::DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity) {
  if (model == nullptr || tickets == nullptr || records == nullptr ||
      actions == nullptr || action_links == nullptr)
    return;
  const std::uint32_t bounded =
      ticket_count < direct_adult_core::kMaxAsynchronousTickets
          ? ticket_count
          : direct_adult_core::kMaxAsynchronousTickets;
  DirectCausalWorldReturnIndex return_index;  // only [0, count) is read
  bool return_index_ready = false, return_index_usable = false;
  for (std::uint32_t slot = 0u; slot < bounded; ++slot) {
    const direct_adult_core::AsynchronousTicket& ticket = tickets[slot];
    if (ticket.ticket_id == 0u || ticket.settled == 0u ||
        model->processed_ticket_by_slot[slot] == ticket.ticket_id)
      continue;
    if (!return_index_ready) {
      return_index_ready = true;
      return_index_usable =
          record_count >= kCausalModelReturnIndexMinimumRecords &&
          causal_model_build_return_index(records, record_count, &return_index);
    }
    const DirectExactHistoryRecord* consequence =
        return_index_usable
            ? causal_model_verified_world_return(ticket, records, return_index)
            : affect_verified_world_return(ticket, records, record_count);
    if (consequence == nullptr) continue;
    const direct_adult_core::DirectActionOccurrence& action = actions[slot];
    if (action.action_ticket_id != ticket.ticket_id ||
        action.state != direct_adult_core::kActionOccurrenceSettled ||
        action.motor_node != ticket.motor_node ||
        action.motor_channel != ticket.motor_channel ||
        action.context_signature != ticket.context_signature ||
        action.emission_tick != ticket.emission_tick ||
        action.expiry_tick < action.emission_tick ||
        action.participant_offset == direct_adult_core::kInvalidIndex ||
        action.participant_offset != slot * action_participant_capacity ||
        action.participant_count == 0u ||
        action.participant_count > action_participant_capacity ||
        action.participant_count > direct_adult_core::kMaxActionParticipationLinks ||
        (action.occurrence_identity_required != 0u &&
         action.occurrence_identity_complete == 0u)) {
      ++model->authority_refusals;
      continue;
    }
    // A real action may integrate several exact sensory roots. Learn the
    // consequence against each root that actually reached the motor closure,
    // rather than rejecting multi-source cognition. Root discovery remains
    // resident-authority based and fails closed per frozen participant.
    std::uint32_t roots[direct_adult_core::kMaxActionParticipationLinks]{};
    std::uint32_t root_count = 0u;
    bool roots_valid = true;
    for (std::uint32_t i = 0u; i < action.participant_count && roots_valid; ++i) {
      const auto& link = action_links[action.participant_offset + i];
      if (link.authority != direct_adult_core::DirectParticipationAuthority::independent_external)
        continue;
      std::uint32_t root = direct_adult_core::kInvalidIndex;
      if (!causal_model_resident_participant_root_channel(
              constructor_state, link, action.emission_tick, &root)) {
        roots_valid = false;
        break;
      }
      bool duplicate = false;
      for (std::uint32_t r = 0u; r < root_count; ++r)
        duplicate |= roots[r] == root;
      if (!duplicate) {
        if (root_count == direct_adult_core::kMaxActionParticipationLinks) {
          roots_valid = false;
          break;
        }
        roots[root_count++] = root;
      }
    }
    if (!roots_valid || root_count == 0u) {
      ++model->authority_refusals;
      continue;
    }
    // Transactional capacity preflight: either every exact root can be
    // represented or this action changes no causal-world relation.
    std::uint32_t new_relations = 0u;
    for (std::uint32_t r = 0u; r < root_count; ++r)
      new_relations += causal_model_find_relation(
          *model, ticket.motor_word, roots[r]) < 0 ? 1u : 0u;
    if (new_relations > kCausalModelRelationCapacity - model->relation_count) {
      ++model->authority_refusals;
      continue;
    }
    bool outcome_capacity_valid = true;
    for (std::uint32_t r = 0u; r < root_count && outcome_capacity_valid; ++r) {
      const std::int32_t existing = causal_model_find_relation(
          *model, ticket.motor_word, roots[r]);
      if (existing < 0) continue;
      const DirectCausalRelation& relation = model->relations[existing];
      bool known = false, empty = false;
      for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b) {
        known |= relation.outcome_counts[b] != 0u &&
                 relation.outcome_values[b] == consequence->value;
        empty |= relation.outcome_counts[b] == 0u;
      }
      outcome_capacity_valid = known || empty;
    }
    if (!outcome_capacity_valid) {
      ++model->authority_refusals;
      continue;
    }
    for (std::uint32_t r = 0u; r < root_count; ++r) {
      const std::uint32_t root_channel = roots[r];
      std::int32_t index =
          causal_model_find_relation(*model, ticket.motor_word, root_channel);
      if (index < 0) {
        index = static_cast<std::int32_t>(model->relation_count);
        DirectCausalRelation fresh{};
        fresh.action_value = ticket.motor_word;
        fresh.root_channel = root_channel;
        model->relations[index] = fresh;
        ++model->relation_count;
      }
      DirectCausalRelation& relation = model->relations[index];
      std::uint32_t standing_best = kCausalModelOutcomeBuckets;
      std::uint32_t standing_best_count = 0u;
      for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b)
        if (relation.outcome_counts[b] > standing_best_count) {
          standing_best_count = relation.outcome_counts[b];
          standing_best = b;
        }
      bool known_bucket = false;
      for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b) {
        if (relation.outcome_counts[b] != 0u &&
            relation.outcome_values[b] == consequence->value) {
          if (standing_best_count > 0u && b != standing_best)
            ++model->revisions;
          ++relation.outcome_counts[b];
          known_bucket = true;
          break;
        }
      }
      if (!known_bucket) {
        std::uint32_t empty_slot = kCausalModelOutcomeBuckets;
        for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b)
          if (relation.outcome_counts[b] == 0u) {
            empty_slot = b;
            break;
          }
        if (empty_slot == kCausalModelOutcomeBuckets) return;
        relation.outcome_values[empty_slot] = consequence->value;
        relation.outcome_counts[empty_slot] = 1u;
        if (relation.observations > 0u) ++model->revisions;
      }
      ++relation.observations;
      if (consequence->resident_tick >= relation.current_outcome_tick) {
        relation.current_outcome_value =
            static_cast<std::uint32_t>(consequence->value);
        relation.current_outcome_tick = consequence->resident_tick;
      }
      model->model_identity = causal_model_fold(
          model->model_identity,
          (static_cast<std::uint64_t>(ticket.motor_word) << 32u) |
              root_channel);
    }
    model->processed_ticket_by_slot[slot] = ticket.ticket_id;
    model->model_identity = causal_model_fold(model->model_identity, ticket.ticket_id);
  }
}

// Majority-vote counterfactual prediction. Refuses unknown pairs, weak
// support, and ambiguous ties: the model predicts only what it actually
// learned, deterministically.
__host__ __device__ inline bool predict_causal_outcome(
    const DirectCausalWorldModel& model, std::uint32_t action_value,
    std::uint32_t root_channel, std::uint32_t* predicted_outcome) {
  if (predicted_outcome == nullptr) return false;
  const std::int32_t index =
      causal_model_find_relation(model, action_value, root_channel);
  if (index < 0) return false;
  const DirectCausalRelation& relation = model.relations[index];
  if (relation.observations < kCausalModelMinimumSupport) return false;
  std::uint32_t best_bucket = kCausalModelOutcomeBuckets;
  std::uint32_t best_count = 0u;
  std::uint32_t runner_up = 0u;
  for (std::uint32_t b = 0u; b < kCausalModelOutcomeBuckets; ++b) {
    if (relation.outcome_counts[b] > best_count) {
      runner_up = best_count;
      best_count = relation.outcome_counts[b];
      best_bucket = b;
    } else if (relation.outcome_counts[b] > runner_up) {
      runner_up = relation.outcome_counts[b];
    }
  }
  if (best_bucket == kCausalModelOutcomeBuckets || best_count == runner_up)
    return false;
  *predicted_outcome = relation.outcome_values[best_bucket];
  return true;
}

}  // namespace substrate::direct_network

#endif
