#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_FAMILY_QUEUE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_FAMILY_QUEUE_CUH

// g.device_family_queue_bucketing (#1574).
// Active resident Recipe IR Occurrences are classified by mechanical
// structural demand and admitted device-side into bounded hardware-aligned
// family queues. Admission, ordering and drain all run on device; the host
// transports pointers and counts and never inspects or composes per-family
// queue state on the dispatch path. Queue placement varies with the
// replaceable backing encoding while queued execution stays byte-identical
// to the canonical generic resident IR path.

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kQueueFamilyCount = 8u;
inline constexpr std::uint32_t kQueueBucketCapacity = 128u;
inline constexpr std::uint32_t kQueueDrainQuantum = 4u;
inline constexpr std::uint32_t kQueueMaxDrainPasses = 64u;

enum class DirectHardwareExecutionClass : std::uint32_t {
  bit = 0u,
  elementwise = 1u,
  contraction = 2u,
  state = 3u,
  semiring = 4u,
  route = 5u,
  structural = 6u,
  micrograph = 7u,
};

static_assert(static_cast<std::uint32_t>(DirectHardwareExecutionClass::bit) ==
                  0u &&
              static_cast<std::uint32_t>(
                  DirectHardwareExecutionClass::elementwise) == 1u &&
              static_cast<std::uint32_t>(
                  DirectHardwareExecutionClass::contraction) == 2u &&
              static_cast<std::uint32_t>(DirectHardwareExecutionClass::state) ==
                  3u &&
              static_cast<std::uint32_t>(
                  DirectHardwareExecutionClass::semiring) == 4u &&
              static_cast<std::uint32_t>(DirectHardwareExecutionClass::route) ==
                  5u &&
              static_cast<std::uint32_t>(
                  DirectHardwareExecutionClass::structural) == 6u &&
              static_cast<std::uint32_t>(
                  DirectHardwareExecutionClass::micrograph) == 7u &&
              kQueueFamilyCount == 8u);

// Mechanical work requirements emitted by resident lowering. Magnitudes size
// work but do not change its class. Exactly one specialized requirement earns
// that homogeneous fast queue; zero/mixed demand uses generic MICROGRAPH.
struct DirectMechanicalExecutionDemand {
  std::uint32_t bit_words;
  std::uint32_t element_count;
  std::uint32_t contraction_terms;
  std::uint32_t state_slots;
  std::uint32_t semiring_terms;
  std::uint32_t route_hops;
  std::uint32_t structural_edits;
  std::uint32_t micrograph_nodes;
};

struct DirectHomogeneousOccurrence {
  DirectMechanicalExecutionDemand demand;
  std::uint64_t sequence;
  std::uint64_t logical_recipe_id;
  std::uint64_t program_identity;
  std::int64_t operand_probe;
  std::uint32_t occurrence;
  std::uint32_t backing_stride;
};

static_assert(std::is_trivial_v<DirectMechanicalExecutionDemand> &&
              std::is_standard_layout_v<DirectMechanicalExecutionDemand> &&
              std::is_trivial_v<DirectHomogeneousOccurrence> &&
              std::is_standard_layout_v<DirectHomogeneousOccurrence>);

__host__ __device__ inline DirectHardwareExecutionClass
classify_homogeneous_execution_class(
    const DirectMechanicalExecutionDemand& demand) {
  const std::uint32_t specialized =
      (demand.bit_words != 0u ? 1u : 0u) +
      (demand.element_count != 0u ? 1u : 0u) +
      (demand.contraction_terms != 0u ? 1u : 0u) +
      (demand.state_slots != 0u ? 1u : 0u) +
      (demand.semiring_terms != 0u ? 1u : 0u) +
      (demand.route_hops != 0u ? 1u : 0u) +
      (demand.structural_edits != 0u ? 1u : 0u);
  if (demand.micrograph_nodes != 0u || specialized != 1u)
    return DirectHardwareExecutionClass::micrograph;
  if (demand.bit_words != 0u) return DirectHardwareExecutionClass::bit;
  if (demand.element_count != 0u)
    return DirectHardwareExecutionClass::elementwise;
  if (demand.contraction_terms != 0u)
    return DirectHardwareExecutionClass::contraction;
  if (demand.state_slots != 0u) return DirectHardwareExecutionClass::state;
  if (demand.semiring_terms != 0u)
    return DirectHardwareExecutionClass::semiring;
  if (demand.route_hops != 0u) return DirectHardwareExecutionClass::route;
  return DirectHardwareExecutionClass::structural;
}

struct DirectFamilyQueueMetrics {
  std::uint64_t offered;
  std::uint64_t admitted;
  std::uint64_t overflow_refusals;
  std::uint64_t executed;
  std::uint64_t host_semantic_dispatches;
  std::uint64_t ordering_roundtrips;
  std::uint64_t host_compositions;
};

struct DirectFamilyQueueEntry {
  std::uint64_t sequence;
  std::uint32_t occurrence;
  std::uint32_t family;
};

struct DirectFamilyQueueBucket {
  DirectFamilyQueueEntry entries[kQueueBucketCapacity];
  std::uint32_t count;
};

struct DirectFamilyQueueDrainStats {
  std::uint32_t passes;
  std::uint32_t served[kQueueMaxDrainPasses][kQueueFamilyCount];
};

struct DirectFamilyQueueBank {
  DirectFamilyQueueBucket buckets[kQueueFamilyCount];
  DirectFamilyQueueMetrics metrics;
  std::uint32_t effective_capacity;
};

static_assert(std::is_trivial_v<DirectFamilyQueueBank> &&
              std::is_standard_layout_v<DirectFamilyQueueBank>);
static_assert(std::is_trivial_v<DirectFamilyQueueDrainStats> &&
              std::is_standard_layout_v<DirectFamilyQueueDrainStats>);

// The dispatch path is host-free: no host ordering roundtrip, no host-side
// composition of queue state, no semantic host dispatch.
__host__ __device__ inline bool direct_family_queue_host_free_law(
    const DirectFamilyQueueMetrics& metrics) {
  return metrics.ordering_roundtrips == 0u && metrics.host_compositions == 0u &&
         metrics.host_semantic_dispatches == 0u;
}

// Positive-control shim: a host-ordered variant must touch this once per
// host-side ordering decision. The lawful device path never calls it.
__host__ __device__ inline void direct_family_queue_note_host_ordering(
    DirectFamilyQueueMetrics* metrics) {
  if (metrics != nullptr) {
    metrics->ordering_roundtrips += 1u;
    metrics->host_compositions += 1u;
  }
}

__device__ inline void direct_family_queue_reset(DirectFamilyQueueBank* bank,
                                                 std::uint32_t capacity) {
  *bank = DirectFamilyQueueBank{};
  bank->effective_capacity =
      capacity == 0u || capacity > kQueueBucketCapacity ? kQueueBucketCapacity
                                                        : capacity;
}

// Mechanical structural demand of an encoded program: op shape and operands
// folded with the replaceable backing-stride selector. Logical recipe identity
// never enters the demand word, and the backing stride never enters execution.
__host__ __device__ inline std::uint64_t queue_demand_word(
    const ResidentRecipeIrProgram& program) {
  std::uint64_t word = static_cast<std::uint64_t>(program.op_count);
  for (std::uint32_t i = 0u;
       i < program.op_count && i < kResidentRecipeIrCapacity; ++i) {
    word = word * 0x9E3779B97F4A7C15ull +
           static_cast<std::uint64_t>(program.instructions[i].op);
    word = word * 0xBF58476D1CE4E5B9ull +
           static_cast<std::uint64_t>(
               static_cast<std::uint32_t>(
                   program.instructions[i].operand_q16));
  }
  word ^= static_cast<std::uint64_t>(program.layout_stride) *
          0xD6E8FEB86659FD93ull;
  return word;
}

__host__ __device__ inline std::uint32_t classify_queue_family(
    const ResidentRecipeIrProgram& program) {
  return static_cast<std::uint32_t>(queue_demand_word(program) &
                                    (kQueueFamilyCount - 1u));
}

__device__ inline bool direct_family_queue_admit_to_family(
    DirectFamilyQueueBank* bank, std::uint32_t occurrence,
    std::uint64_t sequence, std::uint32_t family) {
  if (bank == nullptr || family >= kQueueFamilyCount) return false;
  atomicAdd(reinterpret_cast<unsigned long long*>(&bank->metrics.offered), 1ULL);
  DirectFamilyQueueBucket& bucket = bank->buckets[family];
  const std::uint32_t slot = atomicAdd(&bucket.count, 1u);
  if (slot >= bank->effective_capacity) {
    atomicSub(&bucket.count, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &bank->metrics.overflow_refusals),
              1ULL);
    return false;
  }
  bucket.entries[slot] = DirectFamilyQueueEntry{sequence, occurrence, family};
  atomicAdd(reinterpret_cast<unsigned long long*>(&bank->metrics.admitted), 1ULL);
  return true;
}

__device__ inline bool direct_homogeneous_queue_admit(
    DirectFamilyQueueBank* bank,
    const DirectHomogeneousOccurrence& occurrence) {
  const std::uint32_t family = static_cast<std::uint32_t>(
      classify_homogeneous_execution_class(occurrence.demand));
  return direct_family_queue_admit_to_family(
      bank, occurrence.occurrence, occurrence.sequence, family);
}

__device__ inline bool direct_family_queue_admit(
    DirectFamilyQueueBank* bank, const ResidentRecipeIrProgram* programs,
    std::uint32_t program_count, std::uint32_t occurrence,
    std::uint64_t sequence) {
  if (bank == nullptr || programs == nullptr || occurrence >= program_count)
    return false;
  const std::uint32_t family = classify_queue_family(programs[occurrence]);
  return direct_family_queue_admit_to_family(bank, occurrence, sequence,
                                             family);
}

// The committed in-bucket total order is fixed by (arrival sequence,
// occurrence) alone, so concurrent admission interleaving can never change
// byte-visible queue content.
__device__ inline void direct_family_queue_sort_bucket(
    DirectFamilyQueueBucket* bucket) {
  const std::uint32_t count = bucket->count <= kQueueBucketCapacity
                                  ? bucket->count
                                  : kQueueBucketCapacity;
  for (std::uint32_t i = 1u; i < count; ++i) {
    const DirectFamilyQueueEntry key = bucket->entries[i];
    std::uint32_t j = i;
    while (j > 0u) {
      const DirectFamilyQueueEntry& prior = bucket->entries[j - 1u];
      if (prior.sequence < key.sequence ||
          (prior.sequence == key.sequence &&
           prior.occurrence <= key.occurrence))
        break;
      bucket->entries[j] = bucket->entries[j - 1u];
      --j;
    }
    bucket->entries[j] = key;
  }
}

__global__ void direct_family_queue_order_kernel(DirectFamilyQueueBank* bank) {
  const std::uint32_t family = blockIdx.x * blockDim.x + threadIdx.x;
  if (bank == nullptr || family >= kQueueFamilyCount) return;
  direct_family_queue_sort_bucket(&bank->buckets[family]);
}

// Fair multi-pass drain: every non-empty family serves up to kQueueDrainQuantum
// heads per pass before any family is revisited, so no bucket starves while
// others drain. Emitted heads carry their pass index for liveness receipts.
template <typename Emit>
__device__ inline void direct_family_queue_drain_passes(
    DirectFamilyQueueBank* bank, DirectFamilyQueueDrainStats* stats,
    Emit&& emit) {
  *stats = DirectFamilyQueueDrainStats{};
  bool any = false;
  for (std::uint32_t family = 0u; family < kQueueFamilyCount; ++family)
    if (bank->buckets[family].count != 0u) any = true;
  while (any && stats->passes < kQueueMaxDrainPasses) {
    const std::uint32_t pass = stats->passes;
    any = false;
    for (std::uint32_t family = 0u; family < kQueueFamilyCount; ++family) {
      DirectFamilyQueueBucket& bucket = bank->buckets[family];
      std::uint32_t served = 0u;
      while (served < kQueueDrainQuantum && bucket.count != 0u) {
        const DirectFamilyQueueEntry head = bucket.entries[0u];
        for (std::uint32_t i = 1u; i < bucket.count; ++i)
          bucket.entries[i - 1u] = bucket.entries[i];
        --bucket.count;
        emit(pass, head);
        ++served;
      }
      stats->served[pass][family] = served;
      if (bucket.count != 0u) any = true;
    }
    ++stats->passes;
  }
}

}  // namespace substrate::direct_adult_core

#endif
