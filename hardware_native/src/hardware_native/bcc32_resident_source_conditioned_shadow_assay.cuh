#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32::source_conditioned_shadow_assay {

// This is a raw-contact mechanism assay, not a trust model or a language
// cell.  Source, context, claim and parent-route values are opaque SiteWords
// arriving at a boundary.  The resident path never turns them into names,
// labels, scores or expected answers.
inline constexpr std::uint32_t kFactCapacity = 64u;
inline constexpr std::uint32_t kRelationCapacity = 16u;
inline constexpr std::uint32_t kShadowCapacity = 16u;
inline constexpr std::uint32_t kLineageCapacity = 16u;

enum class EventKind : std::uint8_t {
  advance = 0u,
  testimony = 1u,
  world_observation = 2u,
};

enum class ContactOrigin : std::uint8_t {
  endogenous = 0u,
  external = 1u,
};

struct Event {
  EventKind kind = EventKind::advance;
  ContactOrigin origin = ContactOrigin::endogenous;
  SiteWord source_route = 0u;
  SiteWord parent_route = 0u;
  SiteWord context = 0u;
  SiteWord claim = 0u;
  SiteWord payload = 0u;
  std::uint32_t episode = 0u;
};

// A contact fact is retained independently of the content disposition it
// caused.  It is never removed when a later observation disagrees.
struct BoundaryFact {
  EventKind kind = EventKind::advance;
  ContactOrigin origin = ContactOrigin::endogenous;
  SiteWord source_route = 0u;
  SiteWord parent_route = 0u;
  SiteWord context = 0u;
  SiteWord claim = 0u;
  SiteWord payload = 0u;
  std::uint32_t episode = 0u;
  std::uint32_t occupied = 0u;
};

// A relation is grown from one raw testimony trajectory toward the raw claim
// trajectory.  Its route, closure and residue rails are ordinary matter and
// are intentionally kept separate: a contradiction recarves current route
// matter without erasing the earlier closure rail.
struct RelationMatter {
  SiteWord source_route = 0u;
  SiteWord context = 0u;
  SiteWord claim = 0u;
  SiteWord successor = 0u;
  SiteWord route_matter = 0u;
  SiteWord closure_matter = 0u;
  SiteWord residue_matter = 0u;
  std::uint32_t prepared_epoch = 0u;
  std::uint32_t occupied = 0u;
};

struct PreparedMatter {
  SiteWord source_route = 0u;
  SiteWord context = 0u;
  SiteWord claim = 0u;
  SiteWord successor = 0u;
  std::uint32_t relation_index = 0xffffffffu;
  std::uint32_t due_tick = 0u;
  std::uint32_t active = 0u;
};

// Parent routes are causal ancestry, not source reputation.  A repeated
// parent route can leave another real contact fact, but it cannot create a
// second independent closure rail for the same raw hypothesis.  A distinct
// parent route can.
struct LineageMatter {
  SiteWord context = 0u;
  SiteWord claim = 0u;
  SiteWord observed = 0u;
  SiteWord parent_route = 0u;
  SiteWord closure_matter = 0u;
  std::uint32_t occupied = 0u;
};

// Receipt fields are observer instrumentation only.  The transition path
// never reads them to choose a route or promote a hypothesis.
struct Receipt {
  std::uint32_t external_contacts = 0u;
  std::uint32_t endogenous_contacts = 0u;
  std::uint32_t facts_retained = 0u;
  std::uint32_t fact_capacity_abstentions = 0u;
  std::uint32_t relations_formed = 0u;
  std::uint32_t duplicate_relations = 0u;
  std::uint32_t relation_capacity_abstentions = 0u;
  std::uint32_t shadows_prepared = 0u;
  std::uint32_t shadow_capacity_abstentions = 0u;
  std::uint32_t external_matches = 0u;
  std::uint32_t endogenous_matches = 0u;
  std::uint32_t external_violations = 0u;
  std::uint32_t endogenous_violations = 0u;
  std::uint32_t omissions = 0u;
  std::uint32_t copied_parent_rejections = 0u;
  std::uint32_t independent_lineages = 0u;
};

struct Matter {
  BoundaryFact facts[kFactCapacity]{};
  RelationMatter relations[kRelationCapacity]{};
  PreparedMatter shadows[kShadowCapacity]{};
  LineageMatter lineages[kLineageCapacity]{};

  std::uint32_t tick = 0u;
  std::uint32_t epoch = 1u;
  SiteWord contact_trace = 0u;
  SiteWord closure_trace = 0u;
  SiteWord endogenous_trace = 0u;
  SiteWord unresolved_residue = 0u;
  SiteWord local_reaction = 0u;
  Receipt receipt{};
};

[[nodiscard]] __host__ __device__ inline SiteWord react_words(
    SiteWord left, SiteWord right, std::uint32_t phase) {
  SiteWord left_state = left;
  SiteWord right_state = right;
  apply_edge_block_forward(left_state, right_state, phase & 3u);
  apply_site_word_forward(left_state);
  apply_site_word_forward(right_state);
  return left_state ^ right_state ^ owned_bond_bit(phase & 3u);
}

[[nodiscard]] __host__ __device__ inline bool same_relation(
    const RelationMatter& relation, SiteWord source_route, SiteWord context,
    SiteWord claim) {
  return relation.occupied != 0u && relation.source_route == source_route &&
         relation.context == context && relation.claim == claim;
}

[[nodiscard]] __host__ __device__ inline bool same_fact(
    const BoundaryFact& fact, const Event& event) {
  return fact.occupied != 0u && fact.kind == event.kind &&
         fact.origin == event.origin && fact.source_route == event.source_route &&
         fact.parent_route == event.parent_route && fact.context == event.context &&
         fact.claim == event.claim && fact.payload == event.payload &&
         fact.episode == event.episode;
}

__host__ __device__ inline void retain_fact(Matter& matter, const Event& event) {
  for (std::uint32_t index = 0u; index < kFactCapacity; ++index) {
    if (same_fact(matter.facts[index], event))
      return;
  }
  for (std::uint32_t index = 0u; index < kFactCapacity; ++index) {
    BoundaryFact& fact = matter.facts[index];
    if (fact.occupied != 0u)
      continue;
    fact.kind = event.kind;
    fact.origin = event.origin;
    fact.source_route = event.source_route;
    fact.parent_route = event.parent_route;
    fact.context = event.context;
    fact.claim = event.claim;
    fact.payload = event.payload;
    fact.episode = event.episode;
    fact.occupied = 1u;
    ++matter.receipt.facts_retained;
    return;
  }
  ++matter.receipt.fact_capacity_abstentions;
}

__host__ __device__ inline void form_relation(Matter& matter,
                                              const Event& event) {
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    RelationMatter& relation = matter.relations[index];
    if (!same_relation(relation, event.source_route, event.context,
                       event.claim))
      continue;
    ++matter.receipt.duplicate_relations;
    return;
  }
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    RelationMatter& relation = matter.relations[index];
    if (relation.occupied != 0u)
      continue;
    relation.source_route = event.source_route;
    relation.context = event.context;
    relation.claim = event.claim;
    relation.successor = event.claim;
    relation.route_matter = react_words(event.source_route ^ event.context,
                                        event.claim, matter.epoch);
    relation.occupied = 1u;
    ++matter.receipt.relations_formed;
    return;
  }
  ++matter.receipt.relation_capacity_abstentions;
}

__host__ __device__ inline void record_contact_matter(Matter& matter,
                                                     const Event& event) {
  matter.contact_trace ^= react_words(event.source_route ^ event.context,
                                       event.payload ^ event.claim,
                                       matter.epoch);
  matter.local_reaction ^= react_words(event.claim, event.payload,
                                       matter.epoch);
}

__host__ __device__ inline void leave_omission(Matter& matter,
                                               const PreparedMatter& shadow) {
  const SiteWord rail = channel_bit(kReactiveShift,
                                    shadow.relation_index & 3u);
  matter.unresolved_residue ^= shadow.successor ^ shadow.claim ^ rail;
  ++matter.receipt.omissions;
}

__host__ __device__ inline void leave_violation(
    Matter& matter, const PreparedMatter& shadow, SiteWord observed,
    ContactOrigin origin) {
  RelationMatter& relation = matter.relations[shadow.relation_index];
  const SiteWord rail = channel_bit(kConformationShift,
                                    shadow.relation_index & 3u);
  relation.route_matter ^= react_words(relation.route_matter,
                                       observed ^ rail, matter.epoch);
  relation.residue_matter ^= shadow.successor ^ observed ^ rail;
  matter.unresolved_residue ^= relation.route_matter ^ relation.residue_matter;
  if (origin == ContactOrigin::external)
    ++matter.receipt.external_violations;
  else
    ++matter.receipt.endogenous_violations;
}

__host__ __device__ inline bool admit_lineage(Matter& matter,
                                              const PreparedMatter& shadow,
                                              SiteWord observed,
                                              SiteWord parent_route) {
  for (std::uint32_t index = 0u; index < kLineageCapacity; ++index) {
    const LineageMatter& lineage = matter.lineages[index];
    if (lineage.occupied != 0u && lineage.context == shadow.context &&
        lineage.claim == shadow.claim && lineage.observed == observed &&
        lineage.parent_route == parent_route) {
      ++matter.receipt.copied_parent_rejections;
      return false;
    }
  }
  for (std::uint32_t index = 0u; index < kLineageCapacity; ++index) {
    LineageMatter& lineage = matter.lineages[index];
    if (lineage.occupied != 0u)
      continue;
    lineage.context = shadow.context;
    lineage.claim = shadow.claim;
    lineage.observed = observed;
    lineage.parent_route = parent_route;
    lineage.closure_matter = react_words(shadow.source_route ^ shadow.context,
                                         observed ^ parent_route,
                                         matter.epoch);
    lineage.occupied = 1u;
    ++matter.receipt.independent_lineages;
    return true;
  }
  return false;
}

__host__ __device__ inline void close_shadow(Matter& matter,
                                             const PreparedMatter& shadow,
                                             ContactOrigin origin,
                                             SiteWord parent_route) {
  RelationMatter& relation = matter.relations[shadow.relation_index];
  relation.route_matter ^= react_words(relation.route_matter,
                                       shadow.successor, matter.epoch);
  relation.closure_matter ^= react_words(shadow.successor ^ shadow.context,
                                         parent_route, matter.epoch);
  matter.closure_trace ^= relation.closure_matter ^ shadow.source_route;
  if (origin == ContactOrigin::external) {
    ++matter.receipt.external_matches;
    (void)admit_lineage(matter, shadow, shadow.successor, parent_route);
  } else {
    matter.endogenous_trace ^= relation.closure_matter ^ shadow.source_route;
    ++matter.receipt.endogenous_matches;
  }
}

__host__ __device__ inline void settle_shadows(Matter& matter,
                                               const Event& event) {
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index) {
    PreparedMatter& shadow = matter.shadows[index];
    if (shadow.active == 0u)
      continue;
    if (matter.tick > shadow.due_tick) {
      leave_omission(matter, shadow);
      shadow.active = 0u;
      continue;
    }
    if (event.kind != EventKind::world_observation ||
        event.context != shadow.context || event.claim != shadow.claim ||
        matter.tick != shadow.due_tick)
      continue;
    if (event.payload == shadow.successor)
      close_shadow(matter, shadow, event.origin, event.parent_route);
    else
      leave_violation(matter, shadow, event.payload, event.origin);
    shadow.active = 0u;
  }
}

__host__ __device__ inline void prepare_for_testimony(
    Matter& matter, const Event& event) {
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    RelationMatter& relation = matter.relations[index];
    if (!same_relation(relation, event.source_route, event.context,
                       event.claim) ||
        relation.prepared_epoch == matter.epoch)
      continue;
    relation.prepared_epoch = matter.epoch;
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
    shadow.source_route = relation.source_route;
    shadow.context = relation.context;
    shadow.claim = relation.claim;
    shadow.successor = relation.successor;
    shadow.relation_index = index;
    shadow.due_tick = matter.tick + 1u;
    shadow.active = 1u;
    ++matter.receipt.shadows_prepared;
  }
}

__host__ __device__ inline void apply_testimony(Matter& matter,
                                                const Event& event) {
  retain_fact(matter, event);
  record_contact_matter(matter, event);
  if (event.origin == ContactOrigin::external)
    form_relation(matter, event);
  prepare_for_testimony(matter, event);
  ++matter.epoch;
}

__host__ __device__ inline void apply_world_observation(
    Matter& matter, const Event& event) {
  retain_fact(matter, event);
  record_contact_matter(matter, event);
  settle_shadows(matter, event);
  ++matter.epoch;
}

__host__ __device__ inline void advance_matter(Matter& matter) {
  ++matter.tick;
  apply_site_word_forward(matter.local_reaction);
  Event no_event{};
  settle_shadows(matter, no_event);
  ++matter.epoch;
}

__host__ __device__ inline void apply_event(Matter& matter,
                                            const Event& event) {
  if (event.kind == EventKind::advance) {
    advance_matter(matter);
    return;
  }
  if (event.kind == EventKind::testimony) {
    if (event.origin == ContactOrigin::endogenous)
      ++matter.receipt.endogenous_contacts;
    else
      ++matter.receipt.external_contacts;
    apply_testimony(matter, event);
    return;
  }
  if (event.origin == ContactOrigin::endogenous)
    ++matter.receipt.endogenous_contacts;
  else
    ++matter.receipt.external_contacts;
  apply_world_observation(matter, event);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t count_active_shadows(
    const Matter& matter) {
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < kShadowCapacity; ++index)
    count += matter.shadows[index].active != 0u ? 1u : 0u;
  return count;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t count_lineages(
    const Matter& matter, SiteWord context, SiteWord claim,
    SiteWord observed) {
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < kLineageCapacity; ++index) {
    const LineageMatter& lineage = matter.lineages[index];
    count += lineage.occupied != 0u && lineage.context == context &&
                     lineage.claim == claim && lineage.observed == observed
                 ? 1u
                 : 0u;
  }
  return count;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t count_facts(
    const Matter& matter, ContactOrigin origin) {
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < kFactCapacity; ++index)
    count += matter.facts[index].occupied != 0u &&
                     matter.facts[index].origin == origin
                 ? 1u
                 : 0u;
  return count;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t find_relation(
    const Matter& matter, SiteWord source_route, SiteWord context,
    SiteWord claim) {
  for (std::uint32_t index = 0u; index < kRelationCapacity; ++index) {
    if (same_relation(matter.relations[index], source_route, context, claim))
      return index;
  }
  return 0xffffffffu;
}

__host__ __device__ inline Matter run_trace(const Event* events,
                                            std::uint32_t count) {
  Matter matter{};
  for (std::uint32_t index = 0u; index < count; ++index)
    apply_event(matter, events[index]);
  return matter;
}

__global__ inline void run_trace_kernel(const Event* events,
                                        std::uint32_t count, Matter* after) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || after == nullptr)
    return;
  Matter matter{};
  if (events != nullptr) {
    for (std::uint32_t index = 0u; index < count; ++index)
      apply_event(matter, events[index]);
  }
  *after = matter;
}

}  // namespace substrate::bcc32::source_conditioned_shadow_assay
