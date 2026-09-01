#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

namespace substrate::bcc32::resident_variable_order_surface {

inline constexpr std::uint32_t kRoleCount = resident_roles::kStructuralRoleCount;
inline constexpr std::uint32_t kRoleBits = 6u;
inline constexpr std::uint32_t kMaximumOrder = 8u;
inline constexpr std::uint32_t kInvalidRole = 0xffffffffu;
inline constexpr std::uint64_t kStructureQuantum = 1ull;
inline constexpr std::uint64_t kSupportQuantum = 1ull;
inline constexpr std::uint32_t kBucketWidth = 4u;
inline constexpr std::uint32_t kPrimaryBucketNumerator = 3u;
inline constexpr std::uint32_t kPrimaryBucketDenominator = 4u;
inline constexpr std::uint32_t kMaximumProbeCount = 2u * kBucketWidth;
inline constexpr std::uint32_t kLearningBlockSize = 128u;
inline constexpr std::uint64_t kSecondaryHashSalt = 0xd6e8feb86659fd93ull;

static_assert(kRoleCount == (1u << kRoleBits));
static_assert((kMaximumOrder + 1u) * kRoleBits + 4u < 64u);

struct TransitionCell {
  // Zero is empty. Non-zero keys encode the exact role suffix, its order,
  // and the observed continuation. The table is sparse; no source unit,
  // byte offset, sentence, or authored grammatical label is retained.
  std::uint64_t key = 0u;
  std::uint64_t support_mass = 0u;
};

struct FieldScalars {
  std::uint64_t initial_mass = 0u;
  std::uint64_t free_mass = 0u;
  std::uint64_t structure_mass = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t learned_transitions = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t overflow_events = 0u;
  std::uint64_t table_overflow_events = 0u;
  std::uint64_t mass_exhaustion_events = 0u;
  std::uint64_t invalid_transition_events = 0u;
  std::uint64_t secondary_claims = 0u;
  std::uint64_t support_reservation = 0u;
  std::uint64_t support_reservation_used = 0u;
  std::uint64_t attempted_evidence[kMaximumOrder]{};
  std::uint64_t retained_evidence[kMaximumOrder]{};
  std::uint64_t overflow_by_order[kMaximumOrder]{};
  std::uint32_t occupied_cells = 0u;
  std::uint32_t occupied_by_order[kMaximumOrder]{};
  std::uint32_t maximum_probe_count_observed = 0u;
  std::uint32_t support_reservation_active = 0u;
  std::uint32_t maximum_order = kMaximumOrder;
};

struct FieldView {
  TransitionCell* cells = nullptr;
  std::uint32_t cell_capacity = 0u;
  FieldScalars* scalars = nullptr;
};

struct RoleSequenceBatchView {
  const std::uint32_t* roles = nullptr;
  const std::uint32_t* episode_offsets = nullptr;
  std::uint32_t role_count = 0u;
  std::uint32_t episode_count = 0u;
};

struct UnitRoleSequenceBatchView {
  const std::uint32_t* units = nullptr;
  const std::uint32_t* episode_offsets = nullptr;
  const resident_roles::MutableStructuralRole* unit_roles = nullptr;
  std::uint32_t sequence_count = 0u;
  std::uint32_t episode_count = 0u;
  std::uint32_t unit_count = 0u;
};

struct QueryResult {
  std::uint32_t ready = 0u;
  std::uint32_t next_role = kInvalidRole;
  std::uint32_t matched_order = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t support_mass = 0u;
  std::uint64_t context_mass = 0u;
  std::uint64_t field_revision = 0u;
};

struct AuditResult {
  std::uint64_t cell_structure_mass = 0u;
  std::uint64_t cell_support_mass = 0u;
  std::uint64_t represented_mass = 0u;
  std::uint64_t cell_support_by_order[kMaximumOrder]{};
  std::uint32_t occupied_cells = 0u;
  std::uint32_t occupied_by_order[kMaximumOrder]{};
  std::uint32_t invalid_order_cells = 0u;
  std::uint32_t exact = 0u;
};

struct TableLayout {
  std::uint32_t primary_bucket_count = 0u;
  std::uint32_t secondary_bucket_count = 0u;
  std::uint32_t secondary_begin = 0u;
  std::uint32_t usable_cell_capacity = 0u;
};

// The primary and spill regions are disjoint arrays of fixed-width buckets.
// A key can visit exactly one bucket in each tier, so lookup and insertion are
// bounded by kMaximumProbeCount independent of total table capacity.
[[nodiscard]] __host__ __device__ inline TableLayout table_layout(std::uint32_t cell_capacity) {
  TableLayout layout{};
  const std::uint32_t bucket_count = cell_capacity / kBucketWidth;
  if (bucket_count == 0u)
    return layout;
  if (bucket_count == 1u) {
    layout.primary_bucket_count = 1u;
  } else {
    layout.primary_bucket_count =
        (bucket_count * kPrimaryBucketNumerator) / kPrimaryBucketDenominator;
    if (layout.primary_bucket_count == 0u)
      layout.primary_bucket_count = 1u;
    if (layout.primary_bucket_count >= bucket_count)
      layout.primary_bucket_count = bucket_count - 1u;
    layout.secondary_bucket_count = bucket_count - layout.primary_bucket_count;
  }
  layout.secondary_begin = layout.primary_bucket_count * kBucketWidth;
  layout.usable_cell_capacity = bucket_count * kBucketWidth;
  return layout;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

// History is ordered oldest -> newest. Packing is exact at the supported
// anatomy: 6 bits per learned role, 4 bits for order, and one non-zero bias.
[[nodiscard]] __host__ __device__ inline std::uint64_t transition_key(const std::uint32_t* history,
                                                                      std::uint32_t history_count,
                                                                      std::uint32_t order,
                                                                      std::uint32_t next_role) {
  if (history == nullptr || order == 0u || order > kMaximumOrder || order > history_count ||
      next_role >= kRoleCount)
    return 0u;
  std::uint64_t packed = next_role;
  const std::uint32_t begin = history_count - order;
  for (std::uint32_t index = 0u; index < order; ++index) {
    const std::uint32_t role = history[begin + index];
    if (role >= kRoleCount)
      return 0u;
    packed |= static_cast<std::uint64_t>(role) << (kRoleBits * (index + 1u));
  }
  packed |= static_cast<std::uint64_t>(order) << (kRoleBits * (kMaximumOrder + 1u));
  return packed + 1u;
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

enum class ClaimFailure : std::uint32_t {
  kNone = 0u,
  kInvalid = 1u,
  kTableFull = 2u,
  kMassExhausted = 3u,
};

struct CellClaim {
  TransitionCell* cell = nullptr;
  ClaimFailure failure = ClaimFailure::kNone;
  std::uint32_t probes = 0u;
  std::uint32_t tier = 0u;
};

[[nodiscard]] __host__ __device__ inline std::uint32_t transition_order(std::uint64_t key) {
  if (key == 0u)
    return 0u;
  constexpr std::uint32_t shift = kRoleBits * (kMaximumOrder + 1u);
  return static_cast<std::uint32_t>(((key - 1u) >> shift) & 0xfull);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t bucket_begin(std::uint64_t key,
                                                                    std::uint64_t salt,
                                                                    std::uint32_t bucket_count,
                                                                    std::uint32_t region_begin) {
  return bucket_count == 0u
             ? region_begin
             : region_begin +
                   static_cast<std::uint32_t>(mix64(key ^ salt) % bucket_count) * kBucketWidth;
}

[[nodiscard]] __device__ inline TransitionCell* find_cell(FieldView field, std::uint64_t key) {
  const TableLayout layout = table_layout(field.cell_capacity);
  if (field.cells == nullptr || key == 0u || layout.primary_bucket_count == 0u)
    return nullptr;
  const std::uint32_t primary = bucket_begin(key, 0u, layout.primary_bucket_count, 0u);
  for (std::uint32_t slot = 0u; slot < kBucketWidth; ++slot) {
    TransitionCell* cell = field.cells + primary + slot;
    if (atomic_load_u64(&cell->key) == key)
      return cell;
  }
  if (layout.secondary_bucket_count == 0u)
    return nullptr;
  const std::uint32_t secondary =
      bucket_begin(key, kSecondaryHashSalt, layout.secondary_bucket_count, layout.secondary_begin);
  for (std::uint32_t slot = 0u; slot < kBucketWidth; ++slot) {
    TransitionCell* cell = field.cells + secondary + slot;
    if (atomic_load_u64(&cell->key) == key)
      return cell;
  }
  return nullptr;
}

[[nodiscard]] __device__ inline CellClaim find_or_claim_cell(FieldView field, std::uint64_t key,
                                                             std::uint32_t order) {
  CellClaim result{};
  const TableLayout layout = table_layout(field.cell_capacity);
  if (field.cells == nullptr || field.scalars == nullptr || key == 0u || order == 0u ||
      order > kMaximumOrder || layout.primary_bucket_count == 0u) {
    result.failure = ClaimFailure::kInvalid;
    return result;
  }
  for (std::uint32_t tier = 0u; tier < 2u; ++tier) {
    const std::uint32_t bucket_count =
        tier == 0u ? layout.primary_bucket_count : layout.secondary_bucket_count;
    if (bucket_count == 0u)
      continue;
    const std::uint32_t begin =
        bucket_begin(key, tier == 0u ? 0u : kSecondaryHashSalt, bucket_count,
                     tier == 0u ? 0u : layout.secondary_begin);
    for (std::uint32_t slot = 0u; slot < kBucketWidth; ++slot) {
      ++result.probes;
      TransitionCell* cell = field.cells + begin + slot;
      auto* cell_key = reinterpret_cast<unsigned long long*>(&cell->key);
      const std::uint64_t resident = atomicAdd(cell_key, 0ull);
      if (resident == key) {
        result.cell = cell;
        result.tier = tier;
        return result;
      }
      if (resident != 0u)
        continue;
      if (!reserve_mass(field.scalars, kStructureQuantum)) {
        result.failure = ClaimFailure::kMassExhausted;
        return result;
      }
      const std::uint64_t prior = atomicCAS(cell_key, 0ull, key);
      if (prior == 0u) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->structure_mass),
                  static_cast<unsigned long long>(kStructureQuantum));
        atomicAdd(&field.scalars->occupied_cells, 1u);
        atomicAdd(&field.scalars->occupied_by_order[order - 1u], 1u);
        if (tier != 0u)
          atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->secondary_claims), 1ull);
        result.cell = cell;
        result.tier = tier;
        return result;
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->free_mass),
                static_cast<unsigned long long>(kStructureQuantum));
      if (prior == key) {
        result.cell = cell;
        result.tier = tier;
        return result;
      }
    }
  }
  result.failure = ClaimFailure::kTableFull;
  return result;
}

__device__ inline void record_overflow(FieldView field, std::uint32_t order, ClaimFailure failure) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->overflow_events), 1ull);
  if (order != 0u && order <= kMaximumOrder)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->overflow_by_order[order - 1u]),
              1ull);
  if (failure == ClaimFailure::kTableFull)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->table_overflow_events), 1ull);
  else if (failure == ClaimFailure::kMassExhausted)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->mass_exhaustion_events), 1ull);
  else
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->invalid_transition_events),
              1ull);
}

struct ReinforceResult {
  std::uint32_t retained = 0u;
  std::uint32_t probes = 0u;
};

[[nodiscard]] __device__ inline ReinforceResult reinforce_key(FieldView field, std::uint64_t key,
                                                              std::uint32_t order,
                                                              bool support_reserved);

[[nodiscard]] __device__ inline ReinforceResult reinforce_transition(
    FieldView field, const std::uint32_t* history, std::uint32_t history_count, std::uint32_t order,
    std::uint32_t next_role, bool support_reserved) {
  return reinforce_key(field, transition_key(history, history_count, order, next_role), order,
                       support_reserved);
}

[[nodiscard]] __device__ inline ReinforceResult reinforce_key(FieldView field, std::uint64_t key,
                                                              std::uint32_t order,
                                                              bool support_reserved) {
  ReinforceResult result{};
  if (field.scalars == nullptr)
    return result;
  CellClaim claim = find_or_claim_cell(field, key, order);
  result.probes = claim.probes;
  if (claim.cell == nullptr) {
    record_overflow(field, order, claim.failure);
    return result;
  }
  if (!support_reserved && !reserve_mass(field.scalars, kSupportQuantum)) {
    record_overflow(field, order, ClaimFailure::kMassExhausted);
    return result;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&claim.cell->support_mass),
            static_cast<unsigned long long>(kSupportQuantum));
  if (!support_reserved)
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->support_mass),
              static_cast<unsigned long long>(kSupportQuantum));
  result.retained = 1u;
  return result;
}

__global__ void initialize_field_kernel(FieldView field, std::uint64_t represented_mass,
                                        std::uint32_t maximum_order) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (field.cells != nullptr && index < field.cell_capacity)
    field.cells[index] = TransitionCell{};
  if (blockIdx.x == 0u && threadIdx.x == 0u && field.scalars != nullptr) {
    *field.scalars = FieldScalars{};
    field.scalars->initial_mass = represented_mass;
    field.scalars->free_mass = represented_mass;
    field.scalars->maximum_order =
        maximum_order == 0u || maximum_order > kMaximumOrder ? kMaximumOrder : maximum_order;
  }
}

__global__ void learn_role_sequences_kernel(FieldView field, RoleSequenceBatchView batch) {
  const std::uint32_t episode = blockIdx.x;
  if (episode >= batch.episode_count || batch.roles == nullptr ||
      batch.episode_offsets == nullptr || field.scalars == nullptr)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (end > batch.role_count || begin >= end)
    return;
  __shared__ unsigned long long attempted[kMaximumOrder];
  __shared__ unsigned long long retained[kMaximumOrder];
  __shared__ unsigned long long learned;
  __shared__ std::uint32_t maximum_probes;
  __shared__ std::uint32_t support_reserved;
  if (threadIdx.x < kMaximumOrder) {
    attempted[threadIdx.x] = 0ull;
    retained[threadIdx.x] = 0ull;
  }
  if (threadIdx.x == 0u) {
    learned = 0ull;
    maximum_probes = 0u;
    support_reserved = field.scalars->support_reservation_active;
  }
  __syncthreads();
  for (std::uint32_t position = begin + 1u + threadIdx.x; position < end; position += blockDim.x) {
    const std::uint32_t next_role = batch.roles[position];
    if (next_role >= kRoleCount)
      continue;
    const std::uint32_t available = position - begin;
    const std::uint32_t maximum =
        available < field.scalars->maximum_order ? available : field.scalars->maximum_order;
    for (std::uint32_t order = 1u; order <= maximum; ++order) {
      atomicAdd(&attempted[order - 1u], 1ull);
      const ReinforceResult result = reinforce_transition(field, batch.roles + begin, available,
                                                          order, next_role, support_reserved != 0u);
      if (result.retained != 0u)
        atomicAdd(&retained[order - 1u], 1ull);
      atomicMax(&maximum_probes, result.probes);
    }
    atomicAdd(&learned, 1ull);
  }
  __syncthreads();
  if (threadIdx.x < kMaximumOrder) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&field.scalars->attempted_evidence[threadIdx.x]),
        attempted[threadIdx.x]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->retained_evidence[threadIdx.x]),
              retained[threadIdx.x]);
  }
  if (threadIdx.x == 0u) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->learned_transitions), learned);
    atomicMax(&field.scalars->maximum_probe_count_observed, maximum_probes);
    if (support_reserved != 0u) {
      unsigned long long used = 0ull;
      for (std::uint32_t order = 0u; order < kMaximumOrder; ++order)
        used += retained[order];
      atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->support_reservation_used),
                used * kSupportQuantum);
    }
  }
}

__global__ void learn_unit_role_sequences_kernel(FieldView field, UnitRoleSequenceBatchView batch) {
  const std::uint32_t episode = blockIdx.x;
  if (episode >= batch.episode_count || batch.units == nullptr ||
      batch.episode_offsets == nullptr || batch.unit_roles == nullptr || field.scalars == nullptr)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (end > batch.sequence_count || begin >= end)
    return;
  __shared__ unsigned long long attempted[kMaximumOrder];
  __shared__ unsigned long long retained[kMaximumOrder];
  __shared__ unsigned long long learned;
  __shared__ std::uint32_t maximum_probes;
  __shared__ std::uint32_t support_reserved;
  if (threadIdx.x < kMaximumOrder) {
    attempted[threadIdx.x] = 0ull;
    retained[threadIdx.x] = 0ull;
  }
  if (threadIdx.x == 0u) {
    learned = 0ull;
    maximum_probes = 0u;
    support_reserved = field.scalars->support_reservation_active;
  }
  __syncthreads();
  for (std::uint32_t position = begin + 1u + threadIdx.x; position < end; position += blockDim.x) {
    const std::uint32_t next_unit = batch.units[position];
    if (next_unit >= batch.unit_count)
      continue;
    const std::uint32_t next_role = batch.unit_roles[next_unit].role;
    if (next_role >= kRoleCount)
      continue;
    const std::uint32_t available = position - begin;
    const std::uint32_t maximum =
        available < field.scalars->maximum_order ? available : field.scalars->maximum_order;
    std::uint32_t history[kMaximumOrder]{};
    for (std::uint32_t order = 1u; order <= maximum; ++order) {
      bool valid = true;
      for (std::uint32_t offset = 0u; offset < order; ++offset) {
        const std::uint32_t unit = batch.units[position - order + offset];
        if (unit >= batch.unit_count || batch.unit_roles[unit].role >= kRoleCount) {
          valid = false;
          break;
        }
        history[offset] = batch.unit_roles[unit].role;
      }
      if (valid) {
        atomicAdd(&attempted[order - 1u], 1ull);
        const ReinforceResult result = reinforce_key(
            field, transition_key(history, order, order, next_role), order, support_reserved != 0u);
        if (result.retained != 0u)
          atomicAdd(&retained[order - 1u], 1ull);
        atomicMax(&maximum_probes, result.probes);
      }
    }
    atomicAdd(&learned, 1ull);
  }
  __syncthreads();
  if (threadIdx.x < kMaximumOrder) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&field.scalars->attempted_evidence[threadIdx.x]),
        attempted[threadIdx.x]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->retained_evidence[threadIdx.x]),
              retained[threadIdx.x]);
  }
  if (threadIdx.x == 0u) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->learned_transitions), learned);
    atomicMax(&field.scalars->maximum_probe_count_observed, maximum_probes);
    if (support_reserved != 0u) {
      unsigned long long used = 0ull;
      for (std::uint32_t order = 0u; order < kMaximumOrder; ++order)
        used += retained[order];
      atomicAdd(reinterpret_cast<unsigned long long*>(&field.scalars->support_reservation_used),
                used * kSupportQuantum);
    }
  }
}

__global__ void begin_learning_kernel(FieldView field, std::uint64_t support_upper_bound) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || field.scalars == nullptr)
    return;
  field.scalars->support_reservation = 0u;
  field.scalars->support_reservation_used = 0u;
  field.scalars->support_reservation_active = 0u;
  const std::uint64_t free = field.scalars->free_mass;
  const std::uint64_t structure_upper_bound =
      static_cast<std::uint64_t>(table_layout(field.cell_capacity).usable_cell_capacity) *
      kStructureQuantum;
  if (support_upper_bound != 0u && free >= support_upper_bound &&
      free - support_upper_bound >= structure_upper_bound &&
      reserve_mass(field.scalars, support_upper_bound)) {
    field.scalars->support_reservation = support_upper_bound;
    field.scalars->support_reservation_active = 1u;
  }
}

__global__ void finish_learning_kernel(FieldView field) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && field.scalars != nullptr) {
    if (field.scalars->support_reservation_active != 0u) {
      const std::uint64_t reserved = field.scalars->support_reservation;
      const std::uint64_t used = field.scalars->support_reservation_used;
      field.scalars->support_mass += used;
      field.scalars->free_mass += reserved - used;
      field.scalars->support_reservation = 0u;
      field.scalars->support_reservation_used = 0u;
      field.scalars->support_reservation_active = 0u;
    }
    ++field.scalars->revision;
  }
}

[[nodiscard]] __device__ inline std::uint64_t transition_support(FieldView field,
                                                                 const std::uint32_t* history,
                                                                 std::uint32_t history_count,
                                                                 std::uint32_t order,
                                                                 std::uint32_t next_role) {
  TransitionCell* cell = find_cell(field, transition_key(history, history_count, order, next_role));
  return cell == nullptr ? 0u : atomic_load_u64(&cell->support_mass);
}

[[nodiscard]] __device__ inline QueryResult settle_next(FieldView field,
                                                        const std::uint32_t* history,
                                                        std::uint32_t history_count) {
  QueryResult result{};
  result.field_revision = field.scalars == nullptr ? 0u : field.scalars->revision;
  if (field.scalars == nullptr || history == nullptr || history_count == 0u)
    return result;
  std::uint32_t order =
      history_count < field.scalars->maximum_order ? history_count : field.scalars->maximum_order;
  for (; order != 0u; --order) {
    std::uint64_t total = 0u;
    std::uint64_t best = 0u;
    std::uint32_t winner = kInvalidRole;
    for (std::uint32_t role = 0u; role < kRoleCount; ++role) {
      const std::uint64_t support = transition_support(field, history, history_count, order, role);
      total += support;
      if (support > best || (support == best && support != 0u && role < winner)) {
        best = support;
        winner = role;
      }
    }
    if (total != 0u) {
      result.ready = 1u;
      result.next_role = winner;
      result.matched_order = order;
      result.support_mass = best;
      result.context_mass = total;
      return result;
    }
  }
  return result;
}

// Scores one proposed continuation at the longest resident context. A zero
// candidate support with non-zero context mass is explicit learned rejection;
// it does not silently back off past contradictory longer-order evidence.
[[nodiscard]] __device__ inline QueryResult score_candidate(FieldView field,
                                                            const std::uint32_t* history,
                                                            std::uint32_t history_count,
                                                            std::uint32_t candidate_role) {
  QueryResult result{};
  result.next_role = candidate_role;
  result.field_revision = field.scalars == nullptr ? 0u : field.scalars->revision;
  if (field.scalars == nullptr || history == nullptr || history_count == 0u ||
      candidate_role >= kRoleCount)
    return result;
  std::uint32_t order =
      history_count < field.scalars->maximum_order ? history_count : field.scalars->maximum_order;
  for (; order != 0u; --order) {
    std::uint64_t total = 0u;
    std::uint64_t candidate = 0u;
    for (std::uint32_t role = 0u; role < kRoleCount; ++role) {
      const std::uint64_t support = transition_support(field, history, history_count, order, role);
      total += support;
      if (role == candidate_role)
        candidate = support;
    }
    if (total != 0u) {
      result.ready = candidate != 0u;
      result.matched_order = order;
      result.support_mass = candidate;
      result.context_mass = total;
      return result;
    }
  }
  return result;
}

__global__ void settle_next_kernel(FieldView field, const std::uint32_t* history,
                                   std::uint32_t history_count, QueryResult* result) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && result != nullptr)
    *result = settle_next(field, history, history_count);
}

__global__ void lookup_support_keys_kernel(FieldView field, const std::uint64_t* keys,
                                           std::uint32_t key_count, std::uint64_t* support) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_count || keys == nullptr || support == nullptr)
    return;
  TransitionCell* cell = find_cell(field, keys[index]);
  support[index] = cell == nullptr ? 0u : atomic_load_u64(&cell->support_mass);
}

__global__ void audit_field_kernel(FieldView field, AuditResult* result) {
  if (result == nullptr || field.cells == nullptr)
    return;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < field.cell_capacity;
       index += stride) {
    const std::uint64_t key = atomic_load_u64(&field.cells[index].key);
    if (key == 0u)
      continue;
    atomicAdd(&result->occupied_cells, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&result->cell_structure_mass),
              static_cast<unsigned long long>(kStructureQuantum));
    const std::uint64_t support = atomic_load_u64(&field.cells[index].support_mass);
    atomicAdd(reinterpret_cast<unsigned long long*>(&result->cell_support_mass),
              static_cast<unsigned long long>(support));
    const std::uint32_t order = transition_order(key);
    if (order == 0u || order > kMaximumOrder) {
      atomicAdd(&result->invalid_order_cells, 1u);
      continue;
    }
    atomicAdd(&result->occupied_by_order[order - 1u], 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&result->cell_support_by_order[order - 1u]),
              static_cast<unsigned long long>(support));
  }
}

__global__ void finish_audit_kernel(FieldView field, AuditResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr || field.scalars == nullptr)
    return;
  result->represented_mass =
      field.scalars->free_mass + result->cell_structure_mass + result->cell_support_mass;
  bool exact = result->represented_mass == field.scalars->initial_mass &&
               result->cell_structure_mass == field.scalars->structure_mass &&
               result->cell_support_mass == field.scalars->support_mass &&
               result->occupied_cells == field.scalars->occupied_cells &&
               result->invalid_order_cells == 0u;
  for (std::uint32_t order = 0u; order < kMaximumOrder; ++order) {
    exact = exact && result->occupied_by_order[order] == field.scalars->occupied_by_order[order] &&
            result->cell_support_by_order[order] == field.scalars->retained_evidence[order];
  }
  result->exact = exact ? 1u : 0u;
}

__global__ void lesion_field_kernel(FieldView field) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (field.cells != nullptr && index < field.cell_capacity)
    field.cells[index] = TransitionCell{};
  if (blockIdx.x == 0u && threadIdx.x == 0u && field.scalars != nullptr) {
    const std::uint64_t initial = field.scalars->initial_mass;
    const std::uint32_t maximum_order = field.scalars->maximum_order;
    const std::uint64_t revision = field.scalars->revision + 1u;
    *field.scalars = FieldScalars{};
    field.scalars->initial_mass = initial;
    field.scalars->free_mass = initial;
    field.scalars->maximum_order = maximum_order;
    field.scalars->revision = revision;
  }
}

[[nodiscard]] inline cudaError_t initialize(FieldView field, std::uint64_t represented_mass,
                                            std::uint32_t maximum_order = kMaximumOrder,
                                            cudaStream_t stream = nullptr) {
  if (field.cells == nullptr || field.scalars == nullptr ||
      table_layout(field.cell_capacity).primary_bucket_count == 0u || represented_mass == 0u)
    return cudaErrorInvalidValue;
  constexpr std::uint32_t block = 256u;
  initialize_field_kernel<<<(field.cell_capacity + block - 1u) / block, block, 0u, stream>>>(
      field, represented_mass, maximum_order);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t learn(FieldView field, RoleSequenceBatchView batch,
                                       cudaStream_t stream = nullptr) {
  if (field.cells == nullptr || field.scalars == nullptr || field.cell_capacity == 0u ||
      batch.roles == nullptr || batch.episode_offsets == nullptr || batch.episode_count == 0u)
    return cudaErrorInvalidValue;
  const std::uint64_t support_upper_bound =
      static_cast<std::uint64_t>(batch.role_count) * kMaximumOrder;
  begin_learning_kernel<<<1u, 1u, 0u, stream>>>(field, support_upper_bound);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  learn_role_sequences_kernel<<<batch.episode_count, kLearningBlockSize, 0u, stream>>>(field,
                                                                                       batch);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  finish_learning_kernel<<<1u, 1u, 0u, stream>>>(field);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t learn_units(FieldView field, UnitRoleSequenceBatchView batch,
                                             cudaStream_t stream = nullptr) {
  if (field.cells == nullptr || field.scalars == nullptr || field.cell_capacity == 0u ||
      batch.units == nullptr || batch.episode_offsets == nullptr || batch.unit_roles == nullptr ||
      batch.episode_count == 0u || batch.unit_count == 0u)
    return cudaErrorInvalidValue;
  const std::uint64_t support_upper_bound =
      static_cast<std::uint64_t>(batch.sequence_count) * kMaximumOrder;
  begin_learning_kernel<<<1u, 1u, 0u, stream>>>(field, support_upper_bound);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  learn_unit_role_sequences_kernel<<<batch.episode_count, kLearningBlockSize, 0u, stream>>>(field,
                                                                                            batch);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  finish_learning_kernel<<<1u, 1u, 0u, stream>>>(field);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t query(FieldView field, const std::uint32_t* history,
                                       std::uint32_t history_count, QueryResult* result,
                                       cudaStream_t stream = nullptr) {
  if (history == nullptr || history_count == 0u || result == nullptr)
    return cudaErrorInvalidValue;
  settle_next_kernel<<<1u, 1u, 0u, stream>>>(field, history, history_count, result);
  return cudaGetLastError();
}

[[nodiscard]] inline cudaError_t audit(FieldView field, AuditResult* result,
                                       cudaStream_t stream = nullptr) {
  if (field.cells == nullptr || field.scalars == nullptr || result == nullptr ||
      field.cell_capacity == 0u)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(result, 0, sizeof(AuditResult), stream);
  if (status != cudaSuccess)
    return status;
  constexpr std::uint32_t block = 256u;
  constexpr std::uint32_t maximum_blocks = 4096u;
  std::uint32_t blocks = (field.cell_capacity + block - 1u) / block;
  if (blocks > maximum_blocks)
    blocks = maximum_blocks;
  audit_field_kernel<<<blocks, block, 0u, stream>>>(field, result);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  finish_audit_kernel<<<1u, 1u, 0u, stream>>>(field, result);
  return cudaGetLastError();
}

}  // namespace substrate::bcc32::resident_variable_order_surface
