#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

namespace substrate::bcc32::resident_context_state {

inline constexpr std::uint32_t kLegacyRoleCount = resident_roles::kStructuralRoleCount;
inline constexpr std::uint32_t kNeighborhoodRadius = 4u;
inline constexpr std::uint32_t kNeighborhoodWidth = kNeighborhoodRadius * 2u + 1u;
inline constexpr std::uint32_t kRoleDigitBits = 7u;
inline constexpr std::uint32_t kMaximumProbeCount = 64u;
inline constexpr std::uint32_t kInvalidState = 0xffffffffu;
inline constexpr std::uint32_t kInvalidUnit = 0xffffffffu;
inline constexpr std::uint64_t kTombstoneKey = 1ull;
inline constexpr std::uint64_t kContextQuantum = 1ull;
inline constexpr std::uint64_t kStateQuantum = 1ull;
inline constexpr std::uint64_t kMembershipQuantum = 1ull;
inline constexpr std::uint64_t kTransitionQuantum = 1ull;
inline constexpr std::uint64_t kSupportQuantum = 1ull;

static_assert(kLegacyRoleCount == 64u);
static_assert(kNeighborhoodWidth * kRoleDigitBits == 63u);

// A context key is not a digest. Each of the nine slots stores exactly one
// seven-bit digit: zero for an episode boundary, otherwise legacy_role + 1.
// The center and four incoming/outgoing structural neighbours therefore fit
// losslessly in 63 bits. No unit id, byte, token, source position, grammar
// class, or authored label participates in resident context-state identity.
struct ContextCell {
  std::uint64_t neighborhood_key = 0u;
  std::uint64_t support_mass = 0u;
  std::uint32_t state_id = kInvalidState;
  std::uint32_t reserved = 0u;
};

// Membership is a learned surface attachment to a structural state. The unit
// id is retained only at this output boundary; it never enters the structural
// neighborhood key or state allocation decision.
struct MembershipCell {
  std::uint64_t key = 0u;
  std::uint64_t support_mass = 0u;
};

struct TransitionCell {
  std::uint64_t key = 0u;
  std::uint64_t support_mass = 0u;
};

// This is a recomputable acceleration view over the resident membership
// cells. Lesioning the cache alone cannot remove learned matter; refresh() can
// recreate it exactly from the sparse memberships.
struct UnitBinding {
  std::uint32_t state_id = kInvalidState;
  std::uint32_t alternate_state = kInvalidState;
  std::uint32_t confidence = 0u;
  std::uint32_t distinct_states = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t alternate_support_mass = 0u;
};

struct FieldScalars {
  std::uint64_t initial_mass = 0u;
  std::uint64_t free_mass = 0u;
  std::uint64_t context_mass = 0u;
  std::uint64_t state_mass = 0u;
  std::uint64_t membership_mass = 0u;
  std::uint64_t transition_mass = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t learned_observations = 0u;
  std::uint64_t learned_transitions = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t binding_revisions = 0u;
  std::uint64_t lesion_events = 0u;
  std::uint64_t overflow_events = 0u;
  std::uint64_t context_probe_collisions = 0u;
  std::uint64_t membership_probe_collisions = 0u;
  std::uint64_t transition_probe_collisions = 0u;
  std::uint64_t capacity_collisions = 0u;
  std::uint64_t order2_collision_pairs = 0u;
  std::uint32_t occupied_contexts = 0u;
  std::uint32_t active_states = 0u;
  std::uint32_t issued_states = 0u;
  std::uint32_t occupied_memberships = 0u;
  std::uint32_t occupied_transitions = 0u;
  std::uint32_t ambiguous_units = 0u;
  std::uint32_t minimum_recurrence = 2u;
};

struct FieldView {
  ContextCell* contexts = nullptr;
  MembershipCell* memberships = nullptr;
  TransitionCell* transitions = nullptr;
  UnitBinding* unit_bindings = nullptr;
  FieldScalars* scalars = nullptr;
  std::uint32_t context_capacity = 0u;
  std::uint32_t state_capacity = 0u;
  std::uint32_t membership_capacity = 0u;
  std::uint32_t transition_capacity = 0u;
  std::uint32_t unit_capacity = 0u;
};

// Immutable consumer ABI. Downstream realization can compare opaque state
// ids and resident transition support without learning a lexical identity
// table or acquiring write access to this organ.
struct DeviceView {
  const ContextCell* contexts = nullptr;
  const MembershipCell* memberships = nullptr;
  const TransitionCell* transitions = nullptr;
  const UnitBinding* unit_bindings = nullptr;
  const FieldScalars* scalars = nullptr;
  std::uint32_t context_capacity = 0u;
  std::uint32_t membership_capacity = 0u;
  std::uint32_t transition_capacity = 0u;
  std::uint32_t unit_capacity = 0u;
};

struct LearningWorkspaceView {
  std::uint64_t* primary_ranks = nullptr;
  std::uint64_t* alternate_ranks = nullptr;
  std::uint32_t* state_counts = nullptr;
};

struct UnitSequenceBatchView {
  const std::uint32_t* units = nullptr;
  const std::uint32_t* episode_offsets = nullptr;
  const resident_roles::MutableStructuralRole* unit_roles = nullptr;
  std::uint32_t sequence_count = 0u;
  std::uint32_t episode_count = 0u;
  std::uint32_t unit_count = 0u;
};

struct StateRef {
  std::uint32_t ready = 0u;
  std::uint32_t state_id = kInvalidState;
  std::uint32_t confidence = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t alternate_support_mass = 0u;
};

struct SelectionResult {
  std::uint32_t ready = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t source_state = kInvalidState;
  std::uint32_t selected_state = kInvalidState;
  std::uint32_t selected_unit = kInvalidUnit;
  std::uint32_t considered_candidates = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t field_revision = 0u;
};

struct AuditResult {
  std::uint64_t context_mass = 0u;
  std::uint64_t state_mass = 0u;
  std::uint64_t membership_mass = 0u;
  std::uint64_t transition_mass = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t represented_mass = 0u;
  std::uint32_t occupied_contexts = 0u;
  std::uint32_t active_states = 0u;
  std::uint32_t occupied_memberships = 0u;
  std::uint32_t occupied_transitions = 0u;
  std::uint32_t exact = 0u;
};

[[nodiscard]] __host__ __device__ constexpr DeviceView as_device_view(FieldView field) {
  return {field.contexts,
          field.memberships,
          field.transitions,
          field.unit_bindings,
          field.scalars,
          field.context_capacity,
          field.membership_capacity,
          field.transition_capacity,
          field.unit_capacity};
}

// This mix chooses a sparse table probe only. The full exact key remains in
// every claimed cell and is compared on every lookup; the mixed value is
// never exposed as a context state and never contains a lexical unit id.
[[nodiscard]] __host__ __device__ inline std::uint64_t placement_mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] __host__ __device__ constexpr std::uint64_t role_digit(std::uint32_t role) {
  return role < kLegacyRoleCount ? static_cast<std::uint64_t>(role + 1u) : 0u;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t exact_neighborhood_key(
    const std::uint32_t* roles, std::uint32_t count, std::uint32_t center) {
  if (roles == nullptr || center >= count || roles[center] >= kLegacyRoleCount)
    return 0u;
  std::uint64_t key = 0u;
  for (std::uint32_t slot = 0u; slot < kNeighborhoodWidth; ++slot) {
    const std::int64_t offset =
        static_cast<std::int64_t>(slot) - static_cast<std::int64_t>(kNeighborhoodRadius);
    const std::int64_t index = static_cast<std::int64_t>(center) + offset;
    std::uint64_t digit = 0u;
    if (index >= 0 && index < static_cast<std::int64_t>(count)) {
      const std::uint32_t role = roles[static_cast<std::uint32_t>(index)];
      if (role >= kLegacyRoleCount)
        return 0u;
      digit = role_digit(role);
    }
    key |= digit << (slot * kRoleDigitBits);
  }
  return key;
}

[[nodiscard]] __device__ inline std::uint64_t episode_neighborhood_key(UnitSequenceBatchView batch,
                                                                       std::uint32_t begin,
                                                                       std::uint32_t end,
                                                                       std::uint32_t position) {
  if (batch.units == nullptr || batch.unit_roles == nullptr || position < begin || position >= end)
    return 0u;
  std::uint64_t key = 0u;
  for (std::uint32_t slot = 0u; slot < kNeighborhoodWidth; ++slot) {
    const std::int64_t offset =
        static_cast<std::int64_t>(slot) - static_cast<std::int64_t>(kNeighborhoodRadius);
    const std::int64_t index = static_cast<std::int64_t>(position) + offset;
    std::uint64_t digit = 0u;
    if (index >= static_cast<std::int64_t>(begin) && index < static_cast<std::int64_t>(end)) {
      const std::uint32_t unit = batch.units[static_cast<std::uint32_t>(index)];
      if (unit >= batch.unit_count)
        return 0u;
      const std::uint32_t role = batch.unit_roles[unit].role;
      if (role >= kLegacyRoleCount)
        return 0u;
      digit = role_digit(role);
    }
    key |= digit << (slot * kRoleDigitBits);
  }
  return key;
}

// Radius-two projection used only for collision telemetry. Two distinct
// radius-four states sharing this exact projection are precisely a structural
// collision that the legacy short-context route cannot resolve.
[[nodiscard]] __host__ __device__ inline std::uint64_t order2_projection_key(
    std::uint64_t neighborhood_key) {
  constexpr std::uint32_t first_slot = kNeighborhoodRadius - 2u;
  constexpr std::uint32_t width = 5u;
  constexpr std::uint64_t mask = (1ull << (width * kRoleDigitBits)) - 1ull;
  return (neighborhood_key >> (first_slot * kRoleDigitBits)) & mask;
}

[[nodiscard]] __host__ __device__ constexpr std::uint64_t membership_key(std::uint32_t unit,
                                                                         std::uint32_t state) {
  return (static_cast<std::uint64_t>(unit) + 1ull) << 32u |
         (static_cast<std::uint64_t>(state) + 1ull);
}

[[nodiscard]] __host__ __device__ constexpr std::uint64_t transition_key(std::uint32_t from_state,
                                                                         std::uint32_t to_state) {
  return (static_cast<std::uint64_t>(from_state) + 1ull) << 32u |
         (static_cast<std::uint64_t>(to_state) + 1ull);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t key_high(std::uint64_t key) {
  return static_cast<std::uint32_t>((key >> 32u) - 1ull);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t key_low(std::uint64_t key) {
  return static_cast<std::uint32_t>((key & 0xffffffffull) - 1ull);
}

[[nodiscard]] __device__ inline std::uint64_t atomic_load_u64(const std::uint64_t* address) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(address)),
                   0ull);
}

[[nodiscard]] __device__ inline bool reserve_mass(FieldScalars* scalars, std::uint64_t amount) {
  auto* free_mass = reinterpret_cast<unsigned long long*>(&scalars->free_mass);
  unsigned long long observed = atomicAdd(free_mass, 0ull);
  while (observed >= amount) {
    const unsigned long long prior =
        atomicCAS(free_mass, observed, observed - static_cast<unsigned long long>(amount));
    if (prior == observed)
      return true;
    observed = prior;
  }
  return false;
}

__device__ inline void return_mass(FieldScalars* scalars, std::uint64_t amount) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&scalars->free_mass),
            static_cast<unsigned long long>(amount));
}

[[nodiscard]] __device__ inline std::uint32_t bounded_probe_count(std::uint32_t capacity) {
  return capacity < kMaximumProbeCount ? capacity : kMaximumProbeCount;
}

[[nodiscard]] __device__ inline ContextCell* find_context(FieldView field, std::uint64_t key) {
  if (field.contexts == nullptr || field.context_capacity == 0u || key <= 1u)
    return nullptr;
  const std::uint32_t begin =
      static_cast<std::uint32_t>(placement_mix64(key) % field.context_capacity);
  const std::uint32_t probes = bounded_probe_count(field.context_capacity);
  for (std::uint32_t probe = 0u; probe < probes; ++probe) {
    ContextCell* cell = field.contexts + (begin + probe) % field.context_capacity;
    if (atomic_load_u64(&cell->neighborhood_key) == key)
      return cell;
  }
  return nullptr;
}

[[nodiscard]] __device__ inline ContextCell* find_or_claim_context(FieldView field,
                                                                   std::uint64_t key) {
  if (field.contexts == nullptr || field.scalars == nullptr || field.context_capacity == 0u ||
      key <= 1u)
    return nullptr;
  const std::uint32_t begin =
      static_cast<std::uint32_t>(placement_mix64(key) % field.context_capacity);
  const std::uint32_t probes = bounded_probe_count(field.context_capacity);
  for (std::uint32_t probe = 0u; probe < probes; ++probe) {
    ContextCell* cell = field.contexts + (begin + probe) % field.context_capacity;
    auto* address = reinterpret_cast<unsigned long long*>(&cell->neighborhood_key);
    const std::uint64_t resident = atomicAdd(address, 0ull);
    if (resident == key)
      return cell;
    if (resident != 0u && resident != kTombstoneKey) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->context_probe_collisions),
                1ull);
      continue;
    }
    if (!reserve_mass(field.scalars, kContextQuantum))
      return nullptr;
    const std::uint64_t prior = atomicCAS(address, resident, key);
    if (prior == resident) {
      cell->support_mass = 0u;
      cell->state_id = kInvalidState;
      atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->context_mass),
                static_cast<unsigned long long>(kContextQuantum));
      atomicAdd(&field.scalars->occupied_contexts, 1u);
      return cell;
    }
    return_mass(field.scalars, kContextQuantum);
    if (prior == key)
      return cell;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->capacity_collisions), 1ull);
  return nullptr;
}

template <typename Cell>
[[nodiscard]] __device__ inline Cell* find_sparse_cell(Cell* cells, std::uint32_t capacity,
                                                       std::uint64_t key) {
  if (cells == nullptr || capacity == 0u || key <= 1u)
    return nullptr;
  const std::uint32_t begin = static_cast<std::uint32_t>(placement_mix64(key) % capacity);
  const std::uint32_t probes = bounded_probe_count(capacity);
  for (std::uint32_t probe = 0u; probe < probes; ++probe) {
    Cell* cell = cells + (begin + probe) % capacity;
    if (atomic_load_u64(&cell->key) == key)
      return cell;
  }
  return nullptr;
}

template <typename Cell>
[[nodiscard]] __device__ inline Cell* find_or_claim_sparse_cell(
    Cell* cells, std::uint32_t capacity, std::uint64_t key, FieldScalars* scalars,
    std::uint64_t structure_quantum, std::uint64_t* structure_mass, std::uint32_t* occupied_count,
    std::uint64_t* probe_collisions) {
  if (cells == nullptr || scalars == nullptr || capacity == 0u || key <= 1u)
    return nullptr;
  const std::uint32_t begin = static_cast<std::uint32_t>(placement_mix64(key) % capacity);
  const std::uint32_t probes = bounded_probe_count(capacity);
  for (std::uint32_t probe = 0u; probe < probes; ++probe) {
    Cell* cell = cells + (begin + probe) % capacity;
    auto* address = reinterpret_cast<unsigned long long*>(&cell->key);
    const std::uint64_t resident = atomicAdd(address, 0ull);
    if (resident == key)
      return cell;
    if (resident != 0u && resident != kTombstoneKey) {
      atomicAdd(reinterpret_cast<unsigned long long*>(probe_collisions), 1ull);
      continue;
    }
    if (!reserve_mass(scalars, structure_quantum))
      return nullptr;
    const std::uint64_t prior = atomicCAS(address, resident, key);
    if (prior == resident) {
      cell->support_mass = 0u;
      atomicAdd(reinterpret_cast<unsigned long long*>(structure_mass),
                static_cast<unsigned long long>(structure_quantum));
      atomicAdd(occupied_count, 1u);
      return cell;
    }
    return_mass(scalars, structure_quantum);
    if (prior == key)
      return cell;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&scalars->capacity_collisions), 1ull);
  return nullptr;
}

__device__ inline void reinforce_support(FieldScalars* scalars, std::uint64_t* cell_support) {
  if (!reserve_mass(scalars, kSupportQuantum)) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&scalars->overflow_events), 1ull);
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(cell_support),
            static_cast<unsigned long long>(kSupportQuantum));
  atomicAdd(reinterpret_cast<unsigned long long*>(&scalars->support_mass),
            static_cast<unsigned long long>(kSupportQuantum));
}

__global__ void initialize_field_kernel(FieldView field, LearningWorkspaceView workspace,
                                        std::uint64_t represented_mass,
                                        std::uint32_t minimum_recurrence) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (field.contexts != nullptr && index < field.context_capacity)
    field.contexts[index] = ContextCell{};
  if (field.memberships != nullptr && index < field.membership_capacity)
    field.memberships[index] = MembershipCell{};
  if (field.transitions != nullptr && index < field.transition_capacity)
    field.transitions[index] = TransitionCell{};
  if (field.unit_bindings != nullptr && index < field.unit_capacity)
    field.unit_bindings[index] = UnitBinding{};
  if (workspace.primary_ranks != nullptr && index < field.unit_capacity)
    workspace.primary_ranks[index] = 0u;
  if (workspace.alternate_ranks != nullptr && index < field.unit_capacity)
    workspace.alternate_ranks[index] = 0u;
  if (workspace.state_counts != nullptr && index < field.unit_capacity)
    workspace.state_counts[index] = 0u;
  if (blockIdx.x == 0u && threadIdx.x == 0u && field.scalars != nullptr) {
    *field.scalars = FieldScalars{};
    field.scalars->initial_mass = represented_mass;
    field.scalars->free_mass = represented_mass;
    field.scalars->minimum_recurrence = minimum_recurrence == 0u ? 1u : minimum_recurrence;
  }
}

__global__ void observe_contexts_kernel(FieldView field, UnitSequenceBatchView batch) {
  const std::uint32_t episode = blockIdx.x;
  if (episode >= batch.episode_count || batch.episode_offsets == nullptr ||
      field.scalars == nullptr)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (begin >= end || end > batch.sequence_count)
    return;
  for (std::uint32_t position = begin + threadIdx.x; position < end; position += blockDim.x) {
    const std::uint64_t key = episode_neighborhood_key(batch, begin, end, position);
    ContextCell* cell = find_or_claim_context(field, key);
    if (cell == nullptr)
      continue;
    reinforce_support(field.scalars, &cell->support_mass);
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->learned_observations), 1ull);
  }
}

// State promotion is deterministic device work. It scans resident context
// cells in anatomy order and promotes only recurrent neighborhoods. Collision
// accounting is separated below so the exact audit does not serialize the
// capacity-squared comparison space onto this one ordering thread.
__global__ void promote_contexts_kernel(FieldView field) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || field.scalars == nullptr)
    return;
  const std::uint32_t issued_before = field.scalars->issued_states;
  for (std::uint32_t index = 0u; index < field.context_capacity; ++index)
    field.contexts[index].reserved = 0u;
  for (std::uint32_t index = 0u; index < field.context_capacity; ++index) {
    ContextCell& cell = field.contexts[index];
    if (cell.neighborhood_key <= kTombstoneKey || cell.state_id != kInvalidState ||
        cell.support_mass < field.scalars->minimum_recurrence)
      continue;
    if (field.scalars->issued_states >= field.state_capacity ||
        !reserve_mass(field.scalars, kStateQuantum)) {
      ++field.scalars->capacity_collisions;
      continue;
    }
    cell.state_id = field.scalars->issued_states++;
    cell.reserved = 1u;
    ++field.scalars->active_states;
    field.scalars->state_mass += kStateQuantum;
  }
  if (field.context_capacity != 0u && field.scalars->issued_states != issued_before)
    field.contexts[0].reserved |= 2u;
}

// The original serial definition adds one pair whenever a newly promoted
// context shares its exact radius-two projection with a previously promoted
// context. Evaluating the unordered lower triangle and retaining pairs with at
// least one newly promoted endpoint is exactly the same set, independent of
// resident hash-table order. Each CUDA thread contributes at most one atomic.
__global__ void count_promoted_projection_collisions_kernel(FieldView field) {
  if (field.scalars == nullptr || field.context_capacity == 0u ||
      (field.contexts[0].reserved & 2u) == 0u)
    return;
  std::uint64_t local_pairs = 0u;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < field.context_capacity;
       index += stride) {
    const ContextCell& cell = field.contexts[index];
    if (cell.state_id == kInvalidState || cell.neighborhood_key <= kTombstoneKey)
      continue;
    const std::uint64_t projection = order2_projection_key(cell.neighborhood_key);
    for (std::uint32_t peer_index = 0u; peer_index < index; ++peer_index) {
      const ContextCell& peer = field.contexts[peer_index];
      if (peer.state_id == kInvalidState || peer.neighborhood_key <= kTombstoneKey ||
          ((cell.reserved & 1u) == 0u && (peer.reserved & 1u) == 0u))
        continue;
      if (order2_projection_key(peer.neighborhood_key) == projection)
        ++local_pairs;
    }
  }
  if (local_pairs != 0u)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->order2_collision_pairs),
              static_cast<unsigned long long>(local_pairs));
}

__global__ void clear_promotion_marks_kernel(ContextCell* contexts, std::uint32_t capacity) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (contexts != nullptr && index < capacity)
    contexts[index].reserved = 0u;
}

__global__ void learn_memberships_kernel(FieldView field, UnitSequenceBatchView batch) {
  const std::uint32_t episode = blockIdx.x;
  if (episode >= batch.episode_count || batch.episode_offsets == nullptr ||
      field.scalars == nullptr)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (begin >= end || end > batch.sequence_count)
    return;
  for (std::uint32_t position = begin + threadIdx.x; position < end; position += blockDim.x) {
    const std::uint32_t unit = batch.units[position];
    if (unit >= field.unit_capacity || unit >= batch.unit_count)
      continue;
    ContextCell* context =
        find_context(field, episode_neighborhood_key(batch, begin, end, position));
    if (context == nullptr || context->state_id == kInvalidState)
      continue;
    MembershipCell* cell = find_or_claim_sparse_cell(
        field.memberships, field.membership_capacity, membership_key(unit, context->state_id),
        field.scalars, kMembershipQuantum, &field.scalars->membership_mass,
        &field.scalars->occupied_memberships, &field.scalars->membership_probe_collisions);
    if (cell != nullptr)
      reinforce_support(field.scalars, &cell->support_mass);
  }
}

[[nodiscard]] __host__ __device__ inline std::uint64_t binding_rank(std::uint64_t support,
                                                                    std::uint32_t state) {
  const std::uint64_t bounded = support > 0xffffffffull ? 0xffffffffull : support;
  return (bounded << 32u) | (0xffffffffull - state);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t rank_state(std::uint64_t rank) {
  return static_cast<std::uint32_t>(0xffffffffull - (rank & 0xffffffffull));
}

[[nodiscard]] __host__ __device__ constexpr std::uint64_t rank_support(std::uint64_t rank) {
  return rank >> 32u;
}

__global__ void clear_binding_workspace_kernel(FieldView field, LearningWorkspaceView workspace) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit < field.unit_capacity) {
    workspace.primary_ranks[unit] = 0u;
    workspace.alternate_ranks[unit] = 0u;
    workspace.state_counts[unit] = 0u;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u && field.scalars != nullptr)
    field.scalars->ambiguous_units = 0u;
}

__global__ void aggregate_primary_bindings_kernel(FieldView field,
                                                  LearningWorkspaceView workspace) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= field.membership_capacity)
    return;
  const MembershipCell& cell = field.memberships[index];
  if (cell.key <= kTombstoneKey || cell.support_mass == 0u)
    return;
  const std::uint32_t unit = key_high(cell.key);
  const std::uint32_t state = key_low(cell.key);
  if (unit >= field.unit_capacity)
    return;
  atomicMax(reinterpret_cast<unsigned long long*>(workspace.primary_ranks + unit),
            static_cast<unsigned long long>(binding_rank(cell.support_mass, state)));
  atomicAdd(workspace.state_counts + unit, 1u);
}

__global__ void aggregate_alternate_bindings_kernel(FieldView field,
                                                    LearningWorkspaceView workspace) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= field.membership_capacity)
    return;
  const MembershipCell& cell = field.memberships[index];
  if (cell.key <= kTombstoneKey || cell.support_mass == 0u)
    return;
  const std::uint32_t unit = key_high(cell.key);
  const std::uint32_t state = key_low(cell.key);
  if (unit >= field.unit_capacity || state == rank_state(workspace.primary_ranks[unit]))
    return;
  atomicMax(reinterpret_cast<unsigned long long*>(workspace.alternate_ranks + unit),
            static_cast<unsigned long long>(binding_rank(cell.support_mass, state)));
}

__global__ void finalize_bindings_kernel(FieldView field, LearningWorkspaceView workspace) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= field.unit_capacity)
    return;
  const std::uint64_t primary_rank = workspace.primary_ranks[unit];
  const std::uint64_t alternate_rank = workspace.alternate_ranks[unit];
  const UnitBinding previous = field.unit_bindings[unit];
  UnitBinding next{};
  next.distinct_states = workspace.state_counts[unit];
  if (primary_rank != 0u) {
    next.state_id = rank_state(primary_rank);
    next.support_mass = rank_support(primary_rank);
  }
  if (alternate_rank != 0u) {
    next.alternate_state = rank_state(alternate_rank);
    next.alternate_support_mass = rank_support(alternate_rank);
  }
  if (next.support_mass != 0u && next.support_mass > next.alternate_support_mass) {
    next.confidence = static_cast<std::uint32_t>(
        ((next.support_mass - next.alternate_support_mass) * 255ull) / next.support_mass);
  } else if (next.support_mass != 0u) {
    atomicAdd(&field.scalars->ambiguous_units, 1u);
  }
  if (previous.state_id != kInvalidState && next.state_id != kInvalidState &&
      previous.state_id != next.state_id)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->binding_revisions), 1ull);
  field.unit_bindings[unit] = next;
}

[[nodiscard]] __device__ inline StateRef state_for_unit(DeviceView view, std::uint32_t unit) {
  StateRef result{};
  if (view.unit_bindings == nullptr || unit >= view.unit_capacity)
    return result;
  const UnitBinding binding = view.unit_bindings[unit];
  result.state_id = binding.state_id;
  result.confidence = binding.confidence;
  result.support_mass = binding.support_mass;
  result.alternate_support_mass = binding.alternate_support_mass;
  result.ambiguous =
      binding.support_mass != 0u && binding.support_mass == binding.alternate_support_mass ? 1u
                                                                                           : 0u;
  result.ready = binding.state_id != kInvalidState && result.ambiguous == 0u ? 1u : 0u;
  return result;
}

[[nodiscard]] __device__ inline std::uint64_t transition_support(DeviceView view,
                                                                 std::uint32_t from_state,
                                                                 std::uint32_t to_state) {
  TransitionCell* cell =
      find_sparse_cell(const_cast<TransitionCell*>(view.transitions), view.transition_capacity,
                       transition_key(from_state, to_state));
  return cell == nullptr ? 0u : atomic_load_u64(&cell->support_mass);
}

__global__ void learn_transitions_kernel(FieldView field, UnitSequenceBatchView batch) {
  const std::uint32_t episode = blockIdx.x;
  if (episode >= batch.episode_count || batch.episode_offsets == nullptr ||
      field.scalars == nullptr)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (begin >= end || end > batch.sequence_count)
    return;
  const DeviceView view = as_device_view(field);
  for (std::uint32_t position = begin + threadIdx.x; position + 1u < end; position += blockDim.x) {
    const std::uint32_t from_unit = batch.units[position];
    const std::uint32_t to_unit = batch.units[position + 1u];
    const StateRef from = state_for_unit(view, from_unit);
    const StateRef to = state_for_unit(view, to_unit);
    if (from.ready == 0u || to.ready == 0u)
      continue;
    TransitionCell* cell = find_or_claim_sparse_cell(
        field.transitions, field.transition_capacity, transition_key(from.state_id, to.state_id),
        field.scalars, kTransitionQuantum, &field.scalars->transition_mass,
        &field.scalars->occupied_transitions, &field.scalars->transition_probe_collisions);
    if (cell == nullptr)
      continue;
    reinforce_support(field.scalars, &cell->support_mass);
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->learned_transitions), 1ull);
  }
}

[[nodiscard]] __device__ inline SelectionResult select_continuation(
    DeviceView view, std::uint32_t source_unit, const std::uint32_t* candidate_units,
    std::uint32_t candidate_count) {
  SelectionResult result{};
  result.field_revision = view.scalars == nullptr ? 0u : view.scalars->revision;
  const StateRef source = state_for_unit(view, source_unit);
  result.source_state = source.state_id;
  if (source.ready == 0u || candidate_units == nullptr || candidate_count == 0u)
    return result;
  std::uint64_t best = 0u;
  for (std::uint32_t index = 0u; index < candidate_count; ++index) {
    const std::uint32_t unit = candidate_units[index];
    const StateRef candidate = state_for_unit(view, unit);
    if (candidate.ready == 0u)
      continue;
    ++result.considered_candidates;
    const std::uint64_t support = transition_support(view, source.state_id, candidate.state_id);
    if (support > best) {
      best = support;
      result.selected_unit = unit;
      result.selected_state = candidate.state_id;
      result.ambiguous = 0u;
    } else if (support != 0u && support == best && candidate.state_id != result.selected_state) {
      result.ambiguous = 1u;
    }
  }
  result.support_mass = best;
  result.ready = best != 0u && result.ambiguous == 0u ? 1u : 0u;
  if (result.ready == 0u)
    result.selected_unit = kInvalidUnit;
  return result;
}

__global__ void select_continuation_kernel(DeviceView view, std::uint32_t source_unit,
                                           const std::uint32_t* candidate_units,
                                           std::uint32_t candidate_count, SelectionResult* result) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && result != nullptr)
    *result = select_continuation(view, source_unit, candidate_units, candidate_count);
}

__global__ void audit_field_kernel(FieldView field, AuditResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = AuditResult{};
  if (field.scalars == nullptr)
    return;
  for (std::uint32_t index = 0u; index < field.context_capacity; ++index) {
    const ContextCell& cell = field.contexts[index];
    if (cell.neighborhood_key <= kTombstoneKey)
      continue;
    ++result->occupied_contexts;
    result->context_mass += kContextQuantum;
    result->support_mass += cell.support_mass;
    if (cell.state_id != kInvalidState) {
      ++result->active_states;
      result->state_mass += kStateQuantum;
    }
  }
  for (std::uint32_t index = 0u; index < field.membership_capacity; ++index) {
    const MembershipCell& cell = field.memberships[index];
    if (cell.key <= kTombstoneKey)
      continue;
    ++result->occupied_memberships;
    result->membership_mass += kMembershipQuantum;
    result->support_mass += cell.support_mass;
  }
  for (std::uint32_t index = 0u; index < field.transition_capacity; ++index) {
    const TransitionCell& cell = field.transitions[index];
    if (cell.key <= kTombstoneKey)
      continue;
    ++result->occupied_transitions;
    result->transition_mass += kTransitionQuantum;
    result->support_mass += cell.support_mass;
  }
  result->represented_mass = field.scalars->free_mass + result->context_mass + result->state_mass +
                             result->membership_mass + result->transition_mass +
                             result->support_mass;
  result->exact = result->represented_mass == field.scalars->initial_mass &&
                  result->context_mass == field.scalars->context_mass &&
                  result->state_mass == field.scalars->state_mass &&
                  result->membership_mass == field.scalars->membership_mass &&
                  result->transition_mass == field.scalars->transition_mass &&
                  result->support_mass == field.scalars->support_mass &&
                  result->occupied_contexts == field.scalars->occupied_contexts &&
                  result->active_states == field.scalars->active_states &&
                  result->occupied_memberships == field.scalars->occupied_memberships &&
                  result->occupied_transitions == field.scalars->occupied_transitions;
}

// A focal lesion removes one resident state, all surface memberships that
// depend on it, and every incident transition. Removed quanta return to the
// represented free pool; surviving states and unrelated transitions remain.
__global__ void lesion_state_kernel(FieldView field, std::uint32_t state_id) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || field.scalars == nullptr ||
      state_id == kInvalidState)
    return;
  std::uint64_t returned = 0u;
  for (std::uint32_t index = 0u; index < field.context_capacity; ++index) {
    ContextCell& cell = field.contexts[index];
    if (cell.neighborhood_key <= kTombstoneKey || cell.state_id != state_id)
      continue;
    returned += kContextQuantum + kStateQuantum + cell.support_mass;
    field.scalars->context_mass -= kContextQuantum;
    field.scalars->state_mass -= kStateQuantum;
    field.scalars->support_mass -= cell.support_mass;
    --field.scalars->occupied_contexts;
    --field.scalars->active_states;
    cell.support_mass = 0u;
    cell.state_id = kInvalidState;
    cell.neighborhood_key = kTombstoneKey;
  }
  for (std::uint32_t index = 0u; index < field.membership_capacity; ++index) {
    MembershipCell& cell = field.memberships[index];
    if (cell.key <= kTombstoneKey || key_low(cell.key) != state_id)
      continue;
    returned += kMembershipQuantum + cell.support_mass;
    field.scalars->membership_mass -= kMembershipQuantum;
    field.scalars->support_mass -= cell.support_mass;
    --field.scalars->occupied_memberships;
    cell.support_mass = 0u;
    cell.key = kTombstoneKey;
  }
  for (std::uint32_t index = 0u; index < field.transition_capacity; ++index) {
    TransitionCell& cell = field.transitions[index];
    if (cell.key <= kTombstoneKey ||
        (key_high(cell.key) != state_id && key_low(cell.key) != state_id))
      continue;
    returned += kTransitionQuantum + cell.support_mass;
    field.scalars->transition_mass -= kTransitionQuantum;
    field.scalars->support_mass -= cell.support_mass;
    --field.scalars->occupied_transitions;
    cell.support_mass = 0u;
    cell.key = kTombstoneKey;
  }
  field.scalars->free_mass += returned;
  ++field.scalars->lesion_events;
  ++field.scalars->revision;
}

__global__ void lesion_field_kernel(FieldView field, LearningWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || field.scalars == nullptr)
    return;
  for (std::uint32_t index = 0u; index < field.context_capacity; ++index)
    field.contexts[index] = ContextCell{};
  for (std::uint32_t index = 0u; index < field.membership_capacity; ++index)
    field.memberships[index] = MembershipCell{};
  for (std::uint32_t index = 0u; index < field.transition_capacity; ++index)
    field.transitions[index] = TransitionCell{};
  for (std::uint32_t unit = 0u; unit < field.unit_capacity; ++unit) {
    field.unit_bindings[unit] = UnitBinding{};
    workspace.primary_ranks[unit] = 0u;
    workspace.alternate_ranks[unit] = 0u;
    workspace.state_counts[unit] = 0u;
  }
  const std::uint64_t initial_mass = field.scalars->initial_mass;
  const std::uint64_t revision = field.scalars->revision + 1u;
  const std::uint64_t lesion_events = field.scalars->lesion_events + 1u;
  const std::uint32_t minimum_recurrence = field.scalars->minimum_recurrence;
  *field.scalars = FieldScalars{};
  field.scalars->initial_mass = initial_mass;
  field.scalars->free_mass = initial_mass;
  field.scalars->revision = revision;
  field.scalars->lesion_events = lesion_events;
  field.scalars->minimum_recurrence = minimum_recurrence;
}

__global__ void finish_learning_kernel(FieldScalars* scalars) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && scalars != nullptr)
    ++scalars->revision;
}

[[nodiscard]] inline bool valid_field(FieldView field, LearningWorkspaceView workspace) {
  return field.contexts != nullptr && field.memberships != nullptr &&
         field.transitions != nullptr && field.unit_bindings != nullptr &&
         field.scalars != nullptr && field.context_capacity != 0u && field.state_capacity != 0u &&
         field.membership_capacity != 0u && field.transition_capacity != 0u &&
         field.unit_capacity != 0u && workspace.primary_ranks != nullptr &&
         workspace.alternate_ranks != nullptr && workspace.state_counts != nullptr;
}

[[nodiscard]] inline cudaError_t initialize(FieldView field, LearningWorkspaceView workspace,
                                            std::uint64_t represented_mass,
                                            std::uint32_t minimum_recurrence = 2u,
                                            cudaStream_t stream = nullptr) {
  if (!valid_field(field, workspace) || represented_mass == 0u)
    return cudaErrorInvalidValue;
  std::uint32_t count = field.context_capacity;
  if (field.membership_capacity > count)
    count = field.membership_capacity;
  if (field.transition_capacity > count)
    count = field.transition_capacity;
  if (field.unit_capacity > count)
    count = field.unit_capacity;
  constexpr std::uint32_t block = 256u;
  initialize_field_kernel<<<(count + block - 1u) / block, block, 0u, stream>>>(
      field, workspace, represented_mass, minimum_recurrence);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t refresh_bindings(FieldView field, LearningWorkspaceView workspace,
                                                  cudaStream_t stream = nullptr) {
  if (!valid_field(field, workspace))
    return cudaErrorInvalidValue;
  constexpr std::uint32_t block = 256u;
  clear_binding_workspace_kernel<<<(field.unit_capacity + block - 1u) / block, block, 0u, stream>>>(
      field, workspace);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  aggregate_primary_bindings_kernel<<<(field.membership_capacity + block - 1u) / block, block, 0u,
                                      stream>>>(field, workspace);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  aggregate_alternate_bindings_kernel<<<(field.membership_capacity + block - 1u) / block, block, 0u,
                                        stream>>>(field, workspace);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  finalize_bindings_kernel<<<(field.unit_capacity + block - 1u) / block, block, 0u, stream>>>(
      field, workspace);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t learn(FieldView field, LearningWorkspaceView workspace,
                                       UnitSequenceBatchView batch, cudaStream_t stream = nullptr) {
  if (!valid_field(field, workspace) || batch.units == nullptr ||
      batch.episode_offsets == nullptr || batch.unit_roles == nullptr ||
      batch.sequence_count == 0u || batch.episode_count == 0u || batch.unit_count == 0u ||
      batch.unit_count > field.unit_capacity)
    return cudaErrorInvalidValue;
  observe_contexts_kernel<<<batch.episode_count, 128u, 0u, stream>>>(field, batch);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  FieldView promoted_field = field;
  void* promotion_arguments[] = {&promoted_field};
  status = cudaLaunchKernel(reinterpret_cast<const void*>(promote_contexts_kernel),
                            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, promotion_arguments, 0u,
                            stream);
  if (status != cudaSuccess)
    return status;
  constexpr std::uint32_t context_block = 256u;
  const std::uint32_t context_grid = (field.context_capacity + context_block - 1u) / context_block;
  count_promoted_projection_collisions_kernel<<<context_grid, context_block, 0u, stream>>>(field);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  clear_promotion_marks_kernel<<<context_grid, context_block, 0u, stream>>>(field.contexts,
                                                                            field.context_capacity);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  learn_memberships_kernel<<<batch.episode_count, 128u, 0u, stream>>>(field, batch);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  status = refresh_bindings(field, workspace, stream);
  if (status != cudaSuccess)
    return status;
  learn_transitions_kernel<<<batch.episode_count, 128u, 0u, stream>>>(field, batch);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  // A revision is a completed online resident update, not a host-selected
  // winner. All kernels above execute in this same device stream.
  FieldScalars* revision_scalars = field.scalars;
  void* revision_arguments[] = {&revision_scalars};
  return cudaLaunchKernel(reinterpret_cast<const void*>(finish_learning_kernel),
                          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, revision_arguments, 0u,
                          stream);
}

[[nodiscard]] inline cudaError_t lesion_state(FieldView field, LearningWorkspaceView workspace,
                                              std::uint32_t state_id,
                                              cudaStream_t stream = nullptr) {
  if (!valid_field(field, workspace) || state_id == kInvalidState)
    return cudaErrorInvalidValue;
  FieldView lesioned_field = field;
  std::uint32_t lesioned_state_id = state_id;
  void* lesion_arguments[] = {&lesioned_field, &lesioned_state_id};
  cudaError_t status = cudaLaunchKernel(
      reinterpret_cast<const void*>(lesion_state_kernel), dim3{1u, 1u, 1u},
      dim3{2u, 1u, 1u}, lesion_arguments, 0u, stream);
  if (status != cudaSuccess)
    return status;
  return refresh_bindings(field, workspace, stream);
}

}  // namespace substrate::bcc32::resident_context_state
