#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32::resident_predictive_shadow_assay {

// This is a generic resident relation assay, not a language cell.  The only
// values that enter a relation are raw SiteWords received at a boundary.  A
// relation is grown from temporal adjacency and is later used only to
// precondition another raw SiteWord.  No byte, route, vocabulary, source
// name, expected answer, or semantic target is stored here.
inline constexpr std::uint32_t kRelationCapacity = 8u;
inline constexpr std::uint32_t kShadowCapacity = 8u;
inline constexpr std::uint32_t kOmissionGraceTicks = 0u;

enum class TraceKind : std::uint8_t {
  advance = 0u,
  contact = 1u,
};

enum class ContactOrigin : std::uint8_t {
  endogenous = 0u,
  external = 1u,
};

struct TraceEvent {
  TraceKind kind = TraceKind::advance;
  ContactOrigin origin = ContactOrigin::endogenous;
  SiteWord word = 0u;
};

// The fields are ordinary resident matter.  A relation's raw endpoint words
// are not identities assigned by the host: they are the exact physical words
// that were present at the preceding contacts.  Delay is the elapsed resident
// time between the antecedent and its observed successor.
struct RelationMatter {
  SiteWord context = 0u;
  SiteWord antecedent = 0u;
  SiteWord successor = 0u;
  std::uint32_t delay = 0u;
  std::uint32_t occupied = 0u;
  std::uint32_t external_support = 0u;
};

struct PreparedMatter {
  SiteWord successor = 0u;
  std::uint32_t relation_index = 0xffffffffu;
  std::uint32_t due_tick = 0u;
  std::uint32_t pair_epoch = 0u;
  std::uint32_t active = 0u;
};

// Receipt counters are observer instrumentation only.  None is read by the
// resident transition path to select a route or promote a hypothesis.
struct PredictiveShadowReceipt {
  std::uint32_t external_contacts = 0u;
  std::uint32_t endogenous_contacts = 0u;
  std::uint32_t relations_formed = 0u;
  std::uint32_t duplicate_relations = 0u;
  std::uint32_t relation_capacity_abstentions = 0u;
  std::uint32_t shadows_prepared = 0u;
  std::uint32_t shadow_capacity_abstentions = 0u;
  std::uint32_t active_peak = 0u;
  std::uint32_t external_matches = 0u;
  std::uint32_t endogenous_matches = 0u;
  std::uint32_t external_violations = 0u;
  std::uint32_t endogenous_violations = 0u;
  std::uint32_t omissions = 0u;
  std::uint32_t predecessor_unique = 0u;
  std::uint32_t predecessor_ambiguous = 0u;
  std::uint32_t route_projections = 0u;
  std::uint32_t route_lesion_skips = 0u;
  std::uint32_t route_reactions = 0u;
};

struct PredictiveShadowMatter {
  RelationMatter relations[kRelationCapacity]{};
  PreparedMatter shadows[kShadowCapacity]{};
  std::uint32_t prepared_epoch[kRelationCapacity]{};

  // The two most recent raw contacts form the current physical context.  A
  // contact is not converted into a symbolic token or a semantic label.
  SiteWord previous = 0u;
  SiteWord current = 0u;
  std::uint32_t previous_tick = 0u;
  std::uint32_t current_tick = 0u;
  std::uint32_t history_count = 0u;
  std::uint32_t pair_epoch = 1u;
  std::uint32_t tick = 0u;

  // Test-only physical intervention: a bit removes one currently prepared
  // relation rail from the generic downstream projection.  It is not read by
  // any receipt counter or semantic selector, and remains zero in ordinary
  // resident operation.
  std::uint32_t route_lesion_mask = 0u;

  // These are raw aftermath rails.  A fulfilled preparation closes into the
  // closure trace; an incompatible or omitted preparation leaves residue.
  SiteWord closure_trace = 0u;
  SiteWord unresolved_residue = 0u;
  SiteWord local_reaction = 0u;
  SiteWord projected_route = 0u;
  SiteWord projected_route_reaction = 0u;

  // A predecessor bridge is a read-only reconstruction result.  Ambiguous
  // recontact never chooses one predecessor.
  SiteWord predecessor_context = 0u;
  SiteWord predecessor_antecedent = 0u;

  PredictiveShadowReceipt receipt{};
};

[[nodiscard]] __host__ __device__ inline bool relation_equals(
    const RelationMatter& relation, SiteWord context, SiteWord antecedent,
    SiteWord successor, std::uint32_t delay) {
  return relation.occupied != 0u && relation.context == context &&
         relation.antecedent == antecedent && relation.successor == successor &&
         relation.delay == delay;
}

[[nodiscard]] __host__ __device__ inline bool relation_matches_pair(
    const RelationMatter& relation, const PredictiveShadowMatter& matter) {
  return relation.occupied != 0u && matter.history_count >= 2u &&
         relation.context == matter.previous &&
         relation.antecedent == matter.current;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t active_shadow_count(
    const PredictiveShadowMatter& matter) {
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index)
    count += matter.shadows[index].active != 0u ? 1u : 0u;
  return count;
}

__host__ __device__ inline void form_relation(PredictiveShadowMatter& matter,
                                              SiteWord context,
                                              SiteWord antecedent,
                                              SiteWord successor,
                                              std::uint32_t delay,
                                              ContactOrigin origin) {
  if (delay == 0u)
    return;
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    RelationMatter& relation = matter.relations[index];
    if (!relation_equals(relation, context, antecedent, successor, delay))
      continue;
    ++matter.receipt.duplicate_relations;
    if (origin == ContactOrigin::external)
      relation.external_support = 1u;
    return;
  }
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    RelationMatter& relation = matter.relations[index];
    if (relation.occupied != 0u)
      continue;
    relation.context = context;
    relation.antecedent = antecedent;
    relation.successor = successor;
    relation.delay = delay;
    relation.occupied = 1u;
    relation.external_support = origin == ContactOrigin::external ? 1u : 0u;
    ++matter.receipt.relations_formed;
    return;
  }
  ++matter.receipt.relation_capacity_abstentions;
}

[[nodiscard]] __host__ __device__ inline SiteWord omission_residue_word(
    SiteWord expected, std::uint32_t rail_index) {
  return expected ^ channel_bit(kConformationShift, rail_index & 3u);
}

[[nodiscard]] __host__ __device__ inline SiteWord violation_residue_word(
    SiteWord expected, SiteWord observed, std::uint32_t rail_index) {
  return expected ^ observed ^
         channel_bit(kReactiveShift, rail_index & 3u);
}

// Stateless form of the same local reversible contact chemistry used by
// PredictiveShadowMatter. Canonical resident seams can reuse the proved
// reaction without allocating a second predictor state machine or making a
// host-selected prediction authoritative.
[[nodiscard]] __host__ __device__ inline SiteWord local_contact_reaction_word(
    SiteWord current, SiteWord observed, std::uint32_t pair_epoch) {
  SiteWord source = current;
  SiteWord destination = observed;
  apply_edge_block_forward(source, destination, pair_epoch & 3u);
  apply_site_word_forward(source);
  apply_site_word_forward(destination);
  return source ^ destination;
}

__host__ __device__ inline void leave_omission_residue(
    PredictiveShadowMatter& matter, const PreparedMatter& shadow) {
  // The relation index only selects a physical rail in this finite matter. It
  // is not a semantic error code or a host-provided target.
  matter.unresolved_residue ^=
      omission_residue_word(shadow.successor, shadow.relation_index);
  ++matter.receipt.omissions;
}

__host__ __device__ inline void leave_violation_residue(
    PredictiveShadowMatter& matter, const PreparedMatter& shadow,
    SiteWord observed, ContactOrigin origin) {
  matter.unresolved_residue ^=
      violation_residue_word(shadow.successor, observed,
                             shadow.relation_index);
  if (origin == ContactOrigin::external)
    ++matter.receipt.external_violations;
  else
    ++matter.receipt.endogenous_violations;
}

__host__ __device__ inline void close_shadow(PredictiveShadowMatter& matter,
                                             const PreparedMatter& shadow,
                                             ContactOrigin origin) {
  const SiteWord rail = owned_bond_bit(shadow.relation_index & 3u);
  matter.closure_trace ^= shadow.successor ^ rail;
  if (origin == ContactOrigin::external)
    ++matter.receipt.external_matches;
  else
    ++matter.receipt.endogenous_matches;
}

__host__ __device__ inline void conduct_local_contact_reaction(
    PredictiveShadowMatter& matter, SiteWord observed) {
  // The temporal relation is not a host table of answers.  Raw contact also
  // traverses the ordinary reversible BCC edge/site chemistry, so the assay
  // observes a resident reaction rather than merely comparing integers.
  matter.local_reaction ^=
      local_contact_reaction_word(matter.current, observed, matter.pair_epoch);
}

__host__ __device__ inline void settle_shadows(
    PredictiveShadowMatter& matter, bool has_observation, SiteWord observed,
    ContactOrigin origin) {
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index) {
    PreparedMatter& shadow = matter.shadows[index];
    if (shadow.active == 0u)
      continue;
    if (matter.tick > shadow.due_tick + kOmissionGraceTicks) {
      leave_omission_residue(matter, shadow);
      shadow.active = 0u;
      continue;
    }
    if (!has_observation || matter.tick != shadow.due_tick)
      continue;
    if (observed == shadow.successor)
      close_shadow(matter, shadow, origin);
    else
      leave_violation_residue(matter, shadow, observed, origin);
    shadow.active = 0u;
  }
}

__host__ __device__ inline void prepare_shadows(PredictiveShadowMatter& matter) {
  if (matter.history_count < 2u)
    return;
  for (std::uint32_t relation_index = 0u;
       relation_index < kRelationCapacity; ++relation_index) {
    const RelationMatter& relation = matter.relations[relation_index];
    if (!relation_matches_pair(relation, matter) || relation.delay == 0u ||
        matter.prepared_epoch[relation_index] == matter.pair_epoch)
      continue;
    matter.prepared_epoch[relation_index] = matter.pair_epoch;
    std::uint32_t free_shadow = kShadowCapacity;
    for (std::uint32_t shadow_index = 0u; shadow_index < kShadowCapacity;
         ++shadow_index) {
      if (matter.shadows[shadow_index].active == 0u) {
        free_shadow = shadow_index;
        break;
      }
    }
    if (free_shadow == kShadowCapacity) {
      ++matter.receipt.shadow_capacity_abstentions;
      continue;
    }
    PreparedMatter& shadow = matter.shadows[free_shadow];
    shadow.successor = relation.successor;
    shadow.relation_index = relation_index;
    shadow.due_tick = matter.current_tick + relation.delay;
    shadow.pair_epoch = matter.pair_epoch;
    shadow.active = 1u;
    ++matter.receipt.shadows_prepared;
    const std::uint32_t active = active_shadow_count(matter);
    if (active > matter.receipt.active_peak)
      matter.receipt.active_peak = active;
  }
}

// A live prepared future is allowed to perturb an ordinary downstream rail
// before the corresponding boundary event arrives.  This is deliberately a
// raw physical projection: no candidate is named, ranked by a host, or turned
// into an answer.  Multiple rails collide through the same BCC edge/site
// chemistry used by contact, while a lesion simply removes one participating
// rail for the causal control.
__host__ __device__ inline void project_prepared_routes(
    PredictiveShadowMatter& matter) {
  std::uint32_t active = 0u;
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index)
    active += matter.shadows[index].active != 0u ? 1u : 0u;
  if (active == 0u && matter.closure_trace == 0u &&
      matter.unresolved_residue == 0u)
    return;

  SiteWord projected = matter.local_reaction ^ matter.closure_trace ^
                       matter.unresolved_residue;
  for (std::uint32_t shadow_index = 0u; shadow_index < kShadowCapacity;
       ++shadow_index) {
    const PreparedMatter& shadow = matter.shadows[shadow_index];
    if (shadow.active == 0u)
      continue;
    const std::uint32_t relation_bit = shadow.relation_index & 31u;
    if ((matter.route_lesion_mask & (1u << relation_bit)) != 0u) {
      ++matter.receipt.route_lesion_skips;
      continue;
    }
    SiteWord source = projected;
    SiteWord destination =
        shadow.successor ^ channel_bit(kConformationShift, relation_bit & 3u);
    apply_edge_block_forward(source, destination,
                             (matter.tick + shadow.relation_index) & 3u);
    apply_site_word_forward(source);
    apply_site_word_forward(destination);
    projected ^= source ^ destination;
    ++matter.receipt.route_projections;
  }

  matter.projected_route = projected;
  SiteWord reaction = matter.local_reaction;
  apply_edge_block_forward(reaction, projected, matter.pair_epoch & 3u);
  apply_site_word_forward(reaction);
  apply_site_word_forward(projected);
  matter.projected_route_reaction ^= reaction ^ projected;
  matter.local_reaction ^= reaction ^ projected;
  ++matter.receipt.route_reactions;
}

__host__ __device__ inline void reconstruct_predecessor(
    PredictiveShadowMatter& matter, SiteWord observed) {
  std::uint32_t candidates = 0u;
  std::uint32_t candidate_index = 0xffffffffu;
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    const RelationMatter& relation = matter.relations[index];
    if (relation.occupied == 0u || relation.successor != observed)
      continue;
    ++candidates;
    candidate_index = index;
  }
  if (candidates == 1u) {
    const RelationMatter& relation = matter.relations[candidate_index];
    matter.predecessor_context = relation.context;
    matter.predecessor_antecedent = relation.antecedent;
    ++matter.receipt.predecessor_unique;
  } else if (candidates > 1u) {
    ++matter.receipt.predecessor_ambiguous;
  }
}

__host__ __device__ inline void observe_contact(
    PredictiveShadowMatter& matter, SiteWord observed, ContactOrigin origin) {
  conduct_local_contact_reaction(matter, observed);
  settle_shadows(matter, true, observed, origin);
  reconstruct_predecessor(matter, observed);
  if (origin == ContactOrigin::external)
    ++matter.receipt.external_contacts;
  else
    ++matter.receipt.endogenous_contacts;

  if (matter.history_count >= 2u) {
    const std::uint32_t delay = matter.tick - matter.current_tick;
    form_relation(matter, matter.previous, matter.current, observed, delay,
                  origin);
  }
  if (matter.history_count == 0u) {
    matter.current = observed;
    matter.current_tick = matter.tick;
    matter.history_count = 1u;
  } else if (matter.history_count == 1u) {
    matter.previous = matter.current;
    matter.previous_tick = matter.current_tick;
    matter.current = observed;
    matter.current_tick = matter.tick;
    matter.history_count = 2u;
  } else {
    matter.previous = matter.current;
    matter.previous_tick = matter.current_tick;
    matter.current = observed;
    matter.current_tick = matter.tick;
  }
  ++matter.pair_epoch;
}

__host__ __device__ inline void advance_matter(PredictiveShadowMatter& matter) {
  ++matter.tick;
  apply_site_word_forward(matter.local_reaction);
  settle_shadows(matter, false, 0u, ContactOrigin::endogenous);
  prepare_shadows(matter);
  project_prepared_routes(matter);
}

__host__ __device__ inline void apply_trace_event(
    PredictiveShadowMatter& matter, const TraceEvent& event) {
  if (event.kind == TraceKind::advance)
    advance_matter(matter);
  else
    observe_contact(matter, event.word, event.origin);
}

__host__ __device__ inline void reset_activity_keep_relations(
    PredictiveShadowMatter& matter) {
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index)
    matter.shadows[index] = PreparedMatter{};
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index)
    matter.prepared_epoch[index] = 0u;
  matter.previous = 0u;
  matter.current = 0u;
  matter.previous_tick = 0u;
  matter.current_tick = 0u;
  matter.history_count = 0u;
  matter.pair_epoch = 1u;
  matter.tick = 0u;
  matter.route_lesion_mask = 0u;
  matter.closure_trace = 0u;
  matter.unresolved_residue = 0u;
  matter.local_reaction = 0u;
  matter.projected_route = 0u;
  matter.projected_route_reaction = 0u;
  matter.predecessor_context = 0u;
  matter.predecessor_antecedent = 0u;
}

__host__ __device__ inline PredictiveShadowMatter run_trace_from_state(
    PredictiveShadowMatter matter, const TraceEvent* events,
    std::uint32_t count) {
  for (std::uint32_t index = 0u; index < count; ++index)
    apply_trace_event(matter, events[index]);
  return matter;
}

__host__ __device__ inline PredictiveShadowMatter run_trace(
    const TraceEvent* events, std::uint32_t count) {
  PredictiveShadowMatter matter{};
  return run_trace_from_state(matter, events, count);
}

__global__ inline void run_trace_kernel(const PredictiveShadowMatter* before,
                                        const TraceEvent* events,
                                        std::uint32_t count,
                                        PredictiveShadowMatter* after) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || before == nullptr ||
      after == nullptr)
    return;
  PredictiveShadowMatter matter = *before;
  if (events != nullptr) {
    for (std::uint32_t index = 0u; index < count; ++index)
      apply_trace_event(matter, events[index]);
  }
  *after = matter;
}

}  // namespace substrate::bcc32::resident_predictive_shadow_assay
