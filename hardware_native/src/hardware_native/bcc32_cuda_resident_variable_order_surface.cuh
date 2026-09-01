#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "bcc32_cuda_resident_discourse_plan.cuh"

// A bounded resident bridge from raw equality structure to opaque population
// references.  It contains no byte vocabulary, token identity, grammatical
// role, candidate sentence, semantic score, or renderer.  Raw contacts move a
// conserved support budget into context/continuation bindings.  A query may
// commit one population reference only when the represented context has one
// unambiguous resident continuation; competing continuations cause silence.
namespace bcc32_cuda_resident_variable_order_surface {

namespace discourse_plan = bcc32_cuda_resident_discourse_plan;

constexpr std::uint32_t kMinimumContextOrder = 4u;
constexpr std::uint32_t kMaximumContextOrder = 10u;
constexpr std::uint32_t kMaximumEqualityClasses = 16u;
constexpr std::uint32_t kDefaultEntryCapacity = 1024u;
constexpr std::uint32_t kDefaultPopulationCapacity = 4096u;
constexpr std::uint32_t kInvalidClass = 0xffffffffu;

struct TransitionEntry {
  std::uint64_t context_signature = 0u;
  std::uint64_t population_context_hash = 0u;
  std::uint64_t observational_support = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint64_t lesion_escrow = 0u;
  std::uint64_t structure_mass = 0u;
  std::uint32_t context_order = 0u;
  std::uint32_t next_class = kInvalidClass;
  std::uint32_t next_population_reference = 0u;
  std::uint32_t population_bound = 0u;
  std::uint32_t claimed = 0u;
  std::uint32_t context_populations[kMaximumContextOrder]{};
};

struct FieldScalars {
  std::uint64_t initial_mass = 0u;
  std::uint64_t free_mass = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t accepted_windows = 0u;
  std::uint64_t capacity_rejections = 0u;
  std::uint64_t last_boundary_signature = 0u;
  std::uint32_t occupied_entries = 0u;
  std::uint32_t lesion_revision = 0u;
  std::uint32_t maximum_observed_order = 0u;
  std::uint32_t last_boundary_order = 0u;
  std::uint32_t last_boundary_next_class = kInvalidClass;
  std::uint32_t last_boundary_entry_index = 0u;
  std::uint32_t last_boundary_population_bound = 0u;
  std::uint32_t reserved = 0u;
};

struct FieldView {
  TransitionEntry* entries = nullptr;
  std::uint32_t entry_capacity = 0u;
  FieldScalars* scalars = nullptr;
  std::uint32_t population_capacity = 0u;
};

struct PlanResult {
  std::uint32_t ready = 0u;
  std::uint32_t context_order = 0u;
  std::uint32_t next_class = kInvalidClass;
  std::uint32_t population_reference = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t transition_count = 0u;
  std::uint64_t supporting_mass = 0u;
  std::uint64_t field_revision = 0u;
};

static_assert(std::is_trivially_copyable_v<TransitionEntry>);
static_assert(std::is_trivially_copyable_v<FieldScalars>);
static_assert(std::is_trivially_copyable_v<PlanResult>);

__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

// Canonicalize a raw window by first-occurrence equality.  Byte values never
// enter the stored signature: [9,9,4,9] and [201,201,7,201] are identical.
__host__ __device__ inline std::uint64_t construction_signature(const std::uint8_t* bytes,
                                                                std::uint32_t begin,
                                                                std::uint32_t order) {
  if (bytes == nullptr || order < kMinimumContextOrder || order > kMaximumContextOrder)
    return 0u;
  std::uint64_t signature = static_cast<std::uint64_t>(order) << 56u;
  std::uint8_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint8_t value = bytes[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return 0u;
      representatives[representative_count++] = value;
    }
    signature |= static_cast<std::uint64_t>(identity) << (offset * 4u);
  }
  return signature;
}

__host__ __device__ inline std::uint32_t continuation_class(const std::uint8_t* bytes,
                                                            std::uint32_t begin,
                                                            std::uint32_t order) {
  if (bytes == nullptr || order < kMinimumContextOrder || order > kMaximumContextOrder)
    return kInvalidClass;
  std::uint8_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint8_t value = bytes[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return kInvalidClass;
      representatives[representative_count++] = value;
    }
  }
  const std::uint8_t next = bytes[begin + order];
  for (std::uint32_t identity = 0u; identity < representative_count; ++identity)
    if (representatives[identity] == next)
      return identity;
  return representative_count < kMaximumEqualityClasses ? representative_count : kInvalidClass;
}

__host__ __device__ inline std::uint64_t population_context_hash(const std::uint32_t* populations,
                                                                 std::uint32_t begin,
                                                                 std::uint32_t order) {
  if (populations == nullptr || order < kMinimumContextOrder || order > kMaximumContextOrder)
    return 0u;
  std::uint64_t hash = 1469598103934665603ull;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    hash ^= mix64(static_cast<std::uint64_t>(populations[begin + offset]) + 0x9e3779b97f4a7c15ull);
    hash *= 1099511628211ull;
  }
  hash ^= static_cast<std::uint64_t>(order) << 56u;
  return mix64(hash) | 1ull;
}

__host__ __device__ inline std::uint64_t population_equality_signature(
    const std::uint32_t* populations, std::uint32_t begin, std::uint32_t order) {
  if (populations == nullptr || order < kMinimumContextOrder || order > kMaximumContextOrder)
    return 0u;
  std::uint32_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  std::uint64_t signature = static_cast<std::uint64_t>(order) << 56u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint32_t value = populations[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return 0u;
      representatives[representative_count++] = value;
    }
    signature |= static_cast<std::uint64_t>(identity) << (offset * 4u);
  }
  return signature;
}

__host__ __device__ inline std::uint32_t population_continuation_class(
    const std::uint32_t* populations, std::uint32_t begin, std::uint32_t order) {
  if (populations == nullptr || order < kMinimumContextOrder || order > kMaximumContextOrder)
    return kInvalidClass;
  std::uint32_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint32_t value = populations[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return kInvalidClass;
      representatives[representative_count++] = value;
    }
  }
  const std::uint32_t next = populations[begin + order];
  for (std::uint32_t identity = 0u; identity < representative_count; ++identity)
    if (representatives[identity] == next)
      return identity;
  return representative_count < kMaximumEqualityClasses ? representative_count : kInvalidClass;
}

__device__ inline bool valid(FieldView field) {
  return field.entries != nullptr && field.entry_capacity != 0u && field.scalars != nullptr;
}

__device__ inline TransitionEntry* find_or_claim(FieldView field, std::uint64_t signature,
                                                 std::uint32_t order, std::uint32_t next_class) {
  if (!valid(field) || signature == 0u || next_class == kInvalidClass)
    return nullptr;
  const std::uint32_t start = static_cast<std::uint32_t>(
      mix64(signature ^ (static_cast<std::uint64_t>(next_class) << 32u)) % field.entry_capacity);
  for (std::uint32_t probe = 0u; probe < field.entry_capacity; ++probe) {
    TransitionEntry* entry = field.entries + (start + probe) % field.entry_capacity;
    if (entry->claimed != 0u) {
      if (entry->population_bound == 0u && entry->context_signature == signature &&
          entry->context_order == order && entry->next_class == next_class)
        return entry;
      continue;
    }
    if (field.scalars->free_mass == 0u)
      return nullptr;
    entry->context_signature = signature;
    entry->context_order = order;
    entry->next_class = next_class;
    entry->claimed = 1u;
    entry->structure_mass = 1u;
    --field.scalars->free_mass;
    ++field.scalars->occupied_entries;
    return entry;
  }
  return nullptr;
}

__device__ inline bool same_population_context(const TransitionEntry& entry,
                                               const std::uint32_t* populations,
                                               std::uint32_t begin, std::uint32_t order) {
  if (entry.context_order != order)
    return false;
  for (std::uint32_t offset = 0u; offset < order; ++offset)
    if (entry.context_populations[offset] != populations[begin + offset])
      return false;
  return true;
}

__device__ inline TransitionEntry* find_or_claim_population(
    FieldView field, const std::uint32_t* populations, std::uint32_t begin, std::uint32_t order,
    std::uint32_t next_population, std::uint32_t next_class) {
  const std::uint64_t exact_hash = population_context_hash(populations, begin, order);
  if (!valid(field) || exact_hash == 0u || next_class == kInvalidClass ||
      field.population_capacity == 0u || next_population >= field.population_capacity)
    return nullptr;
  for (std::uint32_t offset = 0u; offset < order; ++offset)
    if (populations[begin + offset] >= field.population_capacity)
      return nullptr;
  const std::uint32_t start = static_cast<std::uint32_t>(
      mix64(exact_hash ^ (static_cast<std::uint64_t>(next_population) << 17u) ^
            0xa0761d6478bd642full) %
      field.entry_capacity);
  for (std::uint32_t probe = 0u; probe < field.entry_capacity; ++probe) {
    TransitionEntry* entry = field.entries + (start + probe) % field.entry_capacity;
    if (entry->claimed != 0u) {
      if (entry->population_bound != 0u && entry->population_context_hash == exact_hash &&
          entry->next_population_reference == next_population &&
          same_population_context(*entry, populations, begin, order))
        return entry;
      continue;
    }
    if (field.scalars->free_mass == 0u)
      return nullptr;
    entry->context_signature = population_equality_signature(populations, begin, order);
    entry->population_context_hash = exact_hash;
    entry->context_order = order;
    entry->next_class = next_class;
    entry->next_population_reference = next_population;
    entry->population_bound = 1u;
    entry->claimed = 1u;
    entry->structure_mass = 1u;
    for (std::uint32_t offset = 0u; offset < order; ++offset)
      entry->context_populations[offset] = populations[begin + offset];
    --field.scalars->free_mass;
    ++field.scalars->occupied_entries;
    return entry;
  }
  return nullptr;
}

__device__ inline void move_observation_mass(FieldScalars* scalars, TransitionEntry* entry) {
  if (scalars == nullptr || entry == nullptr)
    return;
  if (entry->counterevidence != 0u)
    --entry->counterevidence;
  else if (scalars->free_mass != 0u)
    --scalars->free_mass;
  else
    return;
  ++entry->observational_support;
}

__global__ void initialize_field_kernel(FieldView field, std::uint64_t initial_mass) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field))
    return;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index)
    field.entries[index] = TransitionEntry{};
  *field.scalars = FieldScalars{};
  field.scalars->initial_mass = initial_mass;
  field.scalars->free_mass = initial_mass;
}

// Contact boundaries are physical stream boundaries.  No sentence, token, or
// role segmentation is supplied.  Each contact contributes every raw window.
__global__ void assimilate_raw_contact_kernel(FieldView field, const std::uint8_t* bytes,
                                              std::uint32_t byte_count,
                                              std::uint32_t maximum_order = kMaximumContextOrder) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field) || bytes == nullptr ||
      byte_count <= kMinimumContextOrder)
    return;
  const std::uint32_t bounded_order =
      maximum_order < kMaximumContextOrder ? maximum_order : kMaximumContextOrder;
  const std::uint32_t observed_order = bounded_order < byte_count ? bounded_order : byte_count - 1u;
  for (std::uint32_t order = kMinimumContextOrder; order <= bounded_order && order < byte_count;
       ++order) {
    for (std::uint32_t begin = 0u; begin + order < byte_count; ++begin) {
      const std::uint64_t signature = construction_signature(bytes, begin, order);
      const std::uint32_t next_class = continuation_class(bytes, begin, order);
      TransitionEntry* entry = find_or_claim(field, signature, order, next_class);
      if (entry == nullptr) {
        ++field.scalars->capacity_rejections;
        continue;
      }
      const std::uint64_t before = entry->observational_support;
      move_observation_mass(field.scalars, entry);
      if (entry->observational_support != before)
        ++field.scalars->accepted_windows;
      if (begin == 0u && order == observed_order) {
        field.scalars->last_boundary_signature = signature;
        field.scalars->last_boundary_order = order;
        field.scalars->last_boundary_next_class = next_class;
        field.scalars->last_boundary_population_bound = 0u;
      }
    }
  }
  if (observed_order > field.scalars->maximum_observed_order)
    field.scalars->maximum_observed_order = observed_order;
  ++field.scalars->revision;
}

// Opaque population references are produced by resident upstream tissue.  This
// path records their exact overlap and continuation without assigning roles,
// words, or semantic classes to any population.
__global__ void assimilate_population_contact_kernel(
    FieldView field, const std::uint32_t* populations, std::uint32_t population_count,
    std::uint32_t maximum_order = kMaximumContextOrder) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field) || populations == nullptr ||
      population_count <= kMinimumContextOrder)
    return;
  const std::uint32_t bounded_order =
      maximum_order < kMaximumContextOrder ? maximum_order : kMaximumContextOrder;
  const std::uint32_t observed_order =
      bounded_order < population_count ? bounded_order : population_count - 1u;
  for (std::uint32_t order = kMinimumContextOrder;
       order <= bounded_order && order < population_count; ++order) {
    for (std::uint32_t begin = 0u; begin + order < population_count; ++begin) {
      const std::uint32_t next_class = population_continuation_class(populations, begin, order);
      TransitionEntry* entry = find_or_claim_population(field, populations, begin, order,
                                                        populations[begin + order], next_class);
      if (entry == nullptr) {
        ++field.scalars->capacity_rejections;
        continue;
      }
      const std::uint64_t before = entry->observational_support;
      move_observation_mass(field.scalars, entry);
      if (entry->observational_support != before)
        ++field.scalars->accepted_windows;
      if (begin == 0u && order == observed_order) {
        field.scalars->last_boundary_signature = entry->context_signature;
        field.scalars->last_boundary_order = order;
        field.scalars->last_boundary_next_class = next_class;
        field.scalars->last_boundary_entry_index =
            static_cast<std::uint32_t>(entry - field.entries);
        field.scalars->last_boundary_population_bound = 1u;
      }
    }
  }
  if (observed_order > field.scalars->maximum_observed_order)
    field.scalars->maximum_observed_order = observed_order;
  ++field.scalars->revision;
}

__device__ inline bool resolve_population_reference(const std::uint8_t* cue_bytes,
                                                    const std::uint32_t* cue_populations,
                                                    std::uint32_t cue_count, std::uint32_t begin,
                                                    std::uint32_t order, std::uint32_t next_class,
                                                    std::uint32_t* population_reference) {
  if (cue_bytes == nullptr || cue_populations == nullptr || population_reference == nullptr ||
      begin + order > cue_count)
    return false;
  std::uint8_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_populations[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint8_t value = cue_bytes[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return false;
      representatives[identity] = value;
      representative_populations[identity] = cue_populations[begin + offset];
      ++representative_count;
    } else if (representative_populations[identity] != cue_populations[begin + offset]) {
      // Equal raw matter must not be assigned inconsistent resident identity.
      return false;
    }
  }
  if (next_class >= representative_count)
    return false;
  *population_reference = representative_populations[next_class];
  return true;
}

// Query only the longest available resident context.  The field never ranks
// candidate sentences.  Exactly one represented continuation class is needed;
// ambiguity or a novel unbound class yields an empty plan.
__global__ void stage_population_plan_kernel(FieldView field, const std::uint8_t* cue_bytes,
                                             const std::uint32_t* cue_populations,
                                             std::uint32_t cue_count,
                                             discourse_plan::ResidentDiscoursePlanState* plan,
                                             PlanResult* result,
                                             std::uint32_t surface_revision = 0u) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || plan == nullptr || result == nullptr)
    return;
  *result = PlanResult{};
  discourse_plan::clear(plan);
  if (!valid(field) || cue_bytes == nullptr || cue_populations == nullptr ||
      cue_count < kMinimumContextOrder)
    return;

  std::uint32_t order = cue_count < kMaximumContextOrder ? cue_count : kMaximumContextOrder;
  for (;; --order) {
    const std::uint32_t begin = cue_count - order;
    const std::uint64_t signature = construction_signature(cue_bytes, begin, order);
    std::uint32_t selected_class = kInvalidClass;
    std::uint64_t supporting_mass = 0u;
    bool ambiguous = false;
    for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
      const TransitionEntry& entry = field.entries[index];
      if (entry.claimed == 0u || entry.population_bound != 0u ||
          entry.context_signature != signature || entry.context_order != order ||
          entry.observational_support == 0u)
        continue;
      if (selected_class == kInvalidClass)
        selected_class = entry.next_class;
      else if (selected_class != entry.next_class)
        ambiguous = true;
      supporting_mass += entry.observational_support;
    }
    if (!ambiguous && selected_class != kInvalidClass) {
      std::uint32_t population_reference = 0u;
      if (resolve_population_reference(cue_bytes, cue_populations, cue_count, begin, order,
                                       selected_class, &population_reference) &&
          discourse_plan::begin_population_plan(
              plan, field.scalars->revision, surface_revision) &&
          discourse_plan::append_population_step(
              plan, &population_reference, 1u, 0u,
              field.scalars->revision) &&
          discourse_plan::commit_population_plan(plan)) {
        result->ready = 1u;
        result->context_order = order;
        result->next_class = selected_class;
        result->population_reference = population_reference;
        result->output_count = 1u;
        result->supporting_mass = supporting_mass;
        result->field_revision = field.scalars->revision;
      }
      return;
    }
    if (order == kMinimumContextOrder)
      return;
  }
}

__host__ __device__ inline std::uint32_t signature_class_at(std::uint64_t signature,
                                                            std::uint32_t offset) {
  return static_cast<std::uint32_t>((signature >> (offset * 4u)) & 0x0fu);
}

// Source-free structural recall does not compare host candidates.  It accepts only
// one resident transition at the longest context order the field has actually
// experienced, expands its equality classes through an opaque resident anchor
// population, and commits that exact population trajectory.  Multiple longest
// trajectories are an honest ambiguity and remain silent.  This is episodic
// structural recall, not composition of a previously unseen trajectory.
__global__ void stage_source_free_population_plan_kernel(
    FieldView field, const std::uint32_t* anchor_populations, std::uint32_t anchor_count,
    discourse_plan::ResidentDiscoursePlanState* plan, PlanResult* result,
    std::uint32_t surface_revision = 0u) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || plan == nullptr || result == nullptr)
    return;
  *result = PlanResult{};
  discourse_plan::clear(plan);
  if (!valid(field) || anchor_populations == nullptr || anchor_count == 0u ||
      field.scalars->maximum_observed_order < kMinimumContextOrder)
    return;

  const std::uint32_t order = field.scalars->maximum_observed_order;
  const TransitionEntry* selected = nullptr;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    const TransitionEntry& entry = field.entries[index];
    if (entry.claimed == 0u || entry.population_bound != 0u || entry.context_order != order ||
        entry.observational_support == 0u)
      continue;
    if (selected != nullptr)
      return;
    selected = &entry;
  }
  if (selected == nullptr || order + 1u > discourse_plan::kMaxPopulationReferences)
    return;

  std::uint32_t population_references[kMaximumContextOrder + 1u]{};
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint32_t equality_class = signature_class_at(selected->context_signature, offset);
    if (equality_class >= anchor_count)
      return;
    population_references[offset] = anchor_populations[equality_class];
  }
  if (selected->next_class >= anchor_count)
    return;
  population_references[order] = anchor_populations[selected->next_class];

  if (!discourse_plan::begin_population_plan(
          plan, field.scalars->revision, surface_revision) ||
      !discourse_plan::append_population_step(
          plan, population_references, order + 1u, 0u,
          field.scalars->revision) ||
      !discourse_plan::commit_population_plan(plan))
    return;
  result->ready = 1u;
  result->context_order = order;
  result->next_class = selected->next_class;
  result->population_reference = population_references[order];
  result->output_count = order + 1u;
  result->supporting_mass = selected->observational_support;
  result->field_revision = field.scalars->revision;
}

__device__ inline const TransitionEntry* unique_supported_transition(FieldView field,
                                                                     std::uint64_t signature,
                                                                     std::uint32_t order) {
  const TransitionEntry* selected = nullptr;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    const TransitionEntry& entry = field.entries[index];
    if (entry.claimed == 0u || entry.population_bound != 0u ||
        entry.context_signature != signature || entry.context_order != order ||
        entry.observational_support == 0u)
      continue;
    if (selected != nullptr)
      return nullptr;
    selected = &entry;
  }
  return selected;
}

__device__ inline bool resolve_symbolic_continuation(const std::uint8_t* symbolic,
                                                     std::uint32_t begin, std::uint32_t order,
                                                     std::uint32_t next_class,
                                                     std::uint8_t* continuation) {
  if (symbolic == nullptr || continuation == nullptr)
    return false;
  std::uint8_t representatives[kMaximumEqualityClasses]{};
  std::uint32_t representative_count = 0u;
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    const std::uint8_t value = symbolic[begin + offset];
    std::uint32_t identity = 0u;
    while (identity < representative_count && representatives[identity] != value)
      ++identity;
    if (identity == representative_count) {
      if (representative_count == kMaximumEqualityClasses)
        return false;
      representatives[representative_count++] = value;
    }
  }
  if (next_class >= representative_count)
    return false;
  *continuation = representatives[next_class];
  return true;
}

// Compose an unseen population trajectory by walking unique resident
// transition fragments from the most recently admitted physical contact
// boundary.  The kernel receives only opaque resident population anchors; no
// candidate trajectory or expected output is supplied.  Missing, ambiguous,
// or cyclic continuations stop the walk, and a single stored transition is
// insufficient for a composition claim.
__global__ void stage_composed_population_plan_kernel(
    FieldView field, const std::uint32_t* anchor_populations, std::uint32_t anchor_count,
    discourse_plan::ResidentDiscoursePlanState* plan, PlanResult* result,
    std::uint32_t surface_revision = 0u) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || plan == nullptr || result == nullptr)
    return;
  *result = PlanResult{};
  discourse_plan::clear(plan);
  if (!valid(field) || anchor_populations == nullptr || anchor_count == 0u)
    return;

  const std::uint32_t order = field.scalars->last_boundary_order;
  if (order < kMinimumContextOrder || order > kMaximumContextOrder ||
      order + 1u > discourse_plan::kMaxPopulationReferences)
    return;
  const TransitionEntry* first = nullptr;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    const TransitionEntry& entry = field.entries[index];
    if (entry.claimed != 0u && entry.observational_support != 0u &&
        entry.context_signature == field.scalars->last_boundary_signature &&
        entry.context_order == order &&
        entry.next_class == field.scalars->last_boundary_next_class) {
      first = &entry;
      break;
    }
  }
  if (first == nullptr)
    return;

  std::uint8_t symbolic[discourse_plan::kMaxPopulationReferences]{};
  std::uint32_t population_references[discourse_plan::kMaxPopulationReferences]{};
  std::uint64_t seen_signatures[discourse_plan::kMaxPopulationReferences]{};
  std::uint32_t seen_next_classes[discourse_plan::kMaxPopulationReferences]{};
  for (std::uint32_t offset = 0u; offset < order; ++offset) {
    symbolic[offset] =
        static_cast<std::uint8_t>(signature_class_at(first->context_signature, offset));
    if (symbolic[offset] >= anchor_count)
      return;
    population_references[offset] = anchor_populations[symbolic[offset]];
  }
  if (first->next_class >= anchor_count)
    return;
  symbolic[order] = static_cast<std::uint8_t>(first->next_class);
  population_references[order] = anchor_populations[first->next_class];

  std::uint32_t output_count = order + 1u;
  std::uint32_t transition_count = 1u;
  std::uint64_t supporting_mass = first->observational_support;
  seen_signatures[0] = first->context_signature;
  seen_next_classes[0] = first->next_class;
  std::uint32_t final_next_class = first->next_class;

  while (output_count < discourse_plan::kMaxPopulationReferences) {
    const std::uint32_t begin = output_count - order;
    const std::uint64_t signature = construction_signature(symbolic, begin, order);
    const TransitionEntry* next = unique_supported_transition(field, signature, order);
    if (next == nullptr)
      break;
    bool repeated = false;
    for (std::uint32_t index = 0u; index < transition_count; ++index) {
      if (seen_signatures[index] == signature && seen_next_classes[index] == next->next_class) {
        repeated = true;
        break;
      }
    }
    if (repeated)
      break;
    std::uint8_t continuation = 0u;
    if (!resolve_symbolic_continuation(symbolic, begin, order, next->next_class, &continuation) ||
        continuation >= anchor_count)
      break;
    symbolic[output_count] = continuation;
    population_references[output_count] = anchor_populations[continuation];
    seen_signatures[transition_count] = signature;
    seen_next_classes[transition_count] = next->next_class;
    ++output_count;
    ++transition_count;
    supporting_mass += next->observational_support;
    final_next_class = next->next_class;
  }
  if (transition_count < 2u)
    return;

  if (!discourse_plan::begin_population_plan(
          plan, field.scalars->revision, surface_revision) ||
      !discourse_plan::append_population_step(
          plan, population_references, output_count, 0u,
          field.scalars->revision) ||
      !discourse_plan::commit_population_plan(plan))
    return;
  result->ready = 1u;
  result->context_order = order;
  result->next_class = final_next_class;
  result->population_reference = population_references[output_count - 1u];
  result->output_count = output_count;
  result->transition_count = transition_count;
  result->supporting_mass = supporting_mass;
  result->field_revision = field.scalars->revision;
}

__device__ inline const TransitionEntry* unique_bound_population_transition(
    FieldView field, const std::uint32_t* populations, std::uint32_t begin, std::uint32_t order) {
  const std::uint64_t exact_hash = population_context_hash(populations, begin, order);
  const TransitionEntry* selected = nullptr;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    const TransitionEntry& entry = field.entries[index];
    if (entry.claimed == 0u || entry.population_bound == 0u || entry.observational_support == 0u ||
        entry.population_context_hash != exact_hash ||
        !same_population_context(entry, populations, begin, order))
      continue;
    if (selected != nullptr)
      return nullptr;
    selected = &entry;
  }
  return selected;
}

// Join resident population fragments through exact shared context.  Later
// transitions may introduce opaque populations absent from the launch window;
// those identities come only from their stored resident continuation evidence.
__global__ void stage_bound_population_composition_kernel(
    FieldView field, discourse_plan::ResidentDiscoursePlanState* plan, PlanResult* result,
    std::uint32_t surface_revision = 0u) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || plan == nullptr || result == nullptr)
    return;
  *result = PlanResult{};
  discourse_plan::clear(plan);
  if (!valid(field) || field.scalars->last_boundary_population_bound == 0u ||
      field.scalars->last_boundary_entry_index >= field.entry_capacity)
    return;
  const TransitionEntry* first = field.entries + field.scalars->last_boundary_entry_index;
  const std::uint32_t order = first->context_order;
  if (first->claimed == 0u || first->population_bound == 0u || first->observational_support == 0u ||
      order < kMinimumContextOrder || order > kMaximumContextOrder ||
      order + 1u > discourse_plan::kMaxPopulationReferences)
    return;

  std::uint32_t population_references[discourse_plan::kMaxPopulationReferences]{};
  std::uint64_t seen_hashes[discourse_plan::kMaxPopulationReferences]{};
  std::uint32_t seen_next_populations[discourse_plan::kMaxPopulationReferences]{};
  for (std::uint32_t offset = 0u; offset < order; ++offset)
    population_references[offset] = first->context_populations[offset];
  population_references[order] = first->next_population_reference;
  std::uint32_t output_count = order + 1u;
  std::uint32_t transition_count = 1u;
  std::uint64_t supporting_mass = first->observational_support;
  seen_hashes[0] = first->population_context_hash;
  seen_next_populations[0] = first->next_population_reference;

  while (output_count < discourse_plan::kMaxPopulationReferences) {
    const std::uint32_t begin = output_count - order;
    const TransitionEntry* next =
        unique_bound_population_transition(field, population_references, begin, order);
    if (next == nullptr)
      break;
    bool repeated = false;
    for (std::uint32_t index = 0u; index < transition_count; ++index) {
      if (seen_hashes[index] == next->population_context_hash &&
          seen_next_populations[index] == next->next_population_reference) {
        repeated = true;
        break;
      }
    }
    if (repeated)
      break;
    population_references[output_count++] = next->next_population_reference;
    seen_hashes[transition_count] = next->population_context_hash;
    seen_next_populations[transition_count] = next->next_population_reference;
    ++transition_count;
    supporting_mass += next->observational_support;
  }
  if (transition_count < 2u)
    return;

  if (!discourse_plan::begin_population_plan(
          plan, field.scalars->revision, surface_revision) ||
      !discourse_plan::append_population_step(
          plan, population_references, output_count, 0u,
          field.scalars->revision) ||
      !discourse_plan::commit_population_plan(plan))
    return;
  result->ready = 1u;
  result->context_order = order;
  result->next_class = kInvalidClass;
  result->population_reference = population_references[output_count - 1u];
  result->output_count = output_count;
  result->transition_count = transition_count;
  result->supporting_mass = supporting_mass;
  result->field_revision = field.scalars->revision;
}

__global__ void lesion_signature_kernel(FieldView field, std::uint64_t signature,
                                        std::uint32_t order, std::uint64_t* moved_mass) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field))
    return;
  std::uint64_t moved = 0u;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    TransitionEntry& entry = field.entries[index];
    if (entry.claimed == 0u || entry.context_signature != signature || entry.context_order != order)
      continue;
    moved += entry.observational_support;
    entry.lesion_escrow += entry.observational_support;
    entry.observational_support = 0u;
  }
  ++field.scalars->revision;
  ++field.scalars->lesion_revision;
  if (moved_mass != nullptr)
    *moved_mass = moved;
}

__global__ void lesion_entry_kernel(FieldView field, std::uint32_t entry_index,
                                    std::uint64_t* moved_mass) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field) || entry_index >= field.entry_capacity)
    return;
  TransitionEntry& entry = field.entries[entry_index];
  const std::uint64_t moved = entry.observational_support;
  entry.lesion_escrow += moved;
  entry.observational_support = 0u;
  ++field.scalars->revision;
  ++field.scalars->lesion_revision;
  if (moved_mass != nullptr)
    *moved_mass = moved;
}

__global__ void restore_lesions_kernel(FieldView field) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field))
    return;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    TransitionEntry& entry = field.entries[index];
    entry.observational_support += entry.lesion_escrow;
    entry.lesion_escrow = 0u;
  }
  ++field.scalars->revision;
}

__global__ void audit_mass_kernel(FieldView field, std::uint64_t* total) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid(field) || total == nullptr)
    return;
  std::uint64_t mass = field.scalars->free_mass;
  for (std::uint32_t index = 0u; index < field.entry_capacity; ++index) {
    const TransitionEntry& entry = field.entries[index];
    mass += entry.structure_mass + entry.observational_support + entry.counterevidence +
            entry.lesion_escrow;
  }
  *total = mass;
}

}  // namespace bcc32_cuda_resident_variable_order_surface

// The role64 sparse-field surface remains a distinct compatibility/performance
// organ.  Its namespace and language contract are independent of the raw
// equality/population bridge above.
#include "bcc32_cuda_resident_variable_order_sparse_field.cuh"
