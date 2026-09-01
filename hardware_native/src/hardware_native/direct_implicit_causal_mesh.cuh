#ifndef HARDWARE_NATIVE_DIRECT_IMPLICIT_CAUSAL_MESH_CUH
#define HARDWARE_NATIVE_DIRECT_IMPLICIT_CAUSAL_MESH_CUH

#include <cstdint>

#include "hardware_native/direct_dynamic_topology_arena.cuh"

namespace substrate::direct_adult {

#if defined(__CUDACC__)
#define DIRECT_IMPLICIT_HD __host__ __device__
#define DIRECT_IMPLICIT_DEVICE __device__
#else
#define DIRECT_IMPLICIT_HD
#define DIRECT_IMPLICIT_DEVICE
#endif

inline constexpr std::uint32_t kImplicitExceptionDisabled = 1u << 0;
inline constexpr std::uint32_t kImplicitFlagMaterialized = 1u << 1;
inline constexpr std::uint32_t kImplicitFlagPendingMaterialize = 1u << 2;
inline constexpr std::uint32_t kImplicitFamilySkipLastSlotForFirstNode = 1u << 0;
inline constexpr std::uint32_t kMaxImplicitActiveFanout = 4u;
inline constexpr std::int32_t kImplicitParticipationThresholdQ16 = 1 << 10;

struct DirectImplicitCandidate {
  std::uint32_t valid;
  std::uint32_t family;
  std::uint32_t virtual_slot;
  std::uint32_t target;
  std::uint32_t delay;
  std::int32_t conductance_q16;
};

struct DirectImplicitCandidateSet {
  std::uint32_t count;
  DirectImplicitCandidate candidates[kMaxImplicitActiveFanout];
};

struct DirectImplicitParticipationResult {
  bool should_materialize;
  std::uint32_t external_diversity;
  std::uint32_t participation_count;
};

DIRECT_IMPLICIT_HD inline std::uint32_t direct_implicit_target(
    const DirectImplicitFamily& family, std::uint32_t source_node, std::uint32_t virtual_slot) {
  if (source_node < family.node_begin || source_node >= family.node_begin + family.node_count ||
      virtual_slot < family.first_virtual_slot ||
      virtual_slot >= family.first_virtual_slot + family.virtual_slot_count ||
      family.node_count == 0u) {
    return kInvalidIndex;
  }
  const std::uint32_t local = source_node - family.node_begin;
  if ((family.flags & kImplicitFamilySkipLastSlotForFirstNode) != 0u && local == 0u &&
      virtual_slot + 1u == family.local_degree) {
    return kInvalidIndex;
  }
  const std::uint32_t target_local =
      (local + family.chord_stride +
       virtual_slot * (1u + (family.lineage & 3u))) %
      family.node_count;
  return family.node_begin + target_local;
}

DIRECT_IMPLICIT_HD inline std::int32_t direct_implicit_conductance_q16(
    const DirectImplicitFamily& family, std::uint32_t source_node, std::uint32_t virtual_slot,
    std::uint64_t context_signature) {
  const std::uint32_t local = source_node - family.node_begin;
  std::uint64_t mix = context_signature ^ (static_cast<std::uint64_t>(local) << 32) ^
                      (static_cast<std::uint64_t>(virtual_slot) << 24) ^ family.lineage;
  mix ^= mix >> 33;
  mix *= 0xff51afd7ed558ccdull;
  mix ^= mix >> 29;
  const std::int32_t f0 = (mix & 1u) != 0u ? 1 : -1;
  const std::int32_t f1 = (mix & 2u) != 0u ? 1 : -1;
  const std::int32_t f2 = (mix & 4u) != 0u ? 1 : -1;
  const std::int32_t f3 = (mix & 8u) != 0u ? 1 : -1;
  std::int64_t value = family.base_conductance_q16;
  value += static_cast<std::int64_t>(family.coeff_q12[0]) * f0 * 16;
  value += static_cast<std::int64_t>(family.coeff_q12[1]) * f1 * 16;
  value += static_cast<std::int64_t>(family.coeff_q12[2]) * f2 * 16;
  value += static_cast<std::int64_t>(family.coeff_q12[3]) * f3 * 16;
  if (value < (1 << 10))
    value = 1 << 10;
  if (value > (4 << 16))
    value = 4 << 16;
  return static_cast<std::int32_t>(value);
}

DIRECT_IMPLICIT_HD inline std::uint64_t direct_implicit_exception_key(
    std::uint32_t family, std::uint32_t source, std::uint32_t virtual_slot) {
  // 24 bits family (+1 keeps zero as empty), 8 bits virtual slot, 32 bits source.
  if (family >= 0x00ffffffu || virtual_slot > 0xffu)
    return 0u;
  return (static_cast<std::uint64_t>(family + 1u) << 40) |
         (static_cast<std::uint64_t>(virtual_slot) << 32) | source;
}

DIRECT_IMPLICIT_HD inline std::uint32_t direct_implicit_exception_bucket(std::uint64_t key,
                                                                         std::uint32_t capacity) {
  std::uint64_t value = key;
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return static_cast<std::uint32_t>(value) & (capacity - 1u);
}

#if defined(__CUDACC__)
__device__ inline bool lookup_direct_implicit_exception(const DirectBrainV01& brain,
                                                        std::uint32_t family,
                                                        std::uint32_t source,
                                                        std::uint32_t virtual_slot,
                                                        DirectImplicitException* out_exception) {
  const std::uint64_t key = direct_implicit_exception_key(family, source, virtual_slot);
  if (key == 0u || brain.implicit.exception_capacity == 0u)
    return false;
  std::uint32_t slot = direct_implicit_exception_bucket(key, brain.implicit.exception_capacity);
  for (std::uint32_t probe = 0; probe < 64u; ++probe) {
    const DirectImplicitException entry = brain.implicit.exceptions[slot];
    if (entry.key == 0u)
      return false;
    if (entry.key == key) {
      if (out_exception != nullptr)
        *out_exception = entry;
      return true;
    }
    slot = (slot + 1u) & (brain.implicit.exception_capacity - 1u);
  }
  return false;
}

__device__ inline bool direct_implicit_is_materialized(const DirectBrainV01& brain,
                                                       std::uint32_t source,
                                                       std::uint32_t family,
                                                       std::uint32_t virtual_slot) {
  if (source >= brain.node_count)
    return false;
  const DirectNode node = brain.nodes[source];
  std::uint32_t route = node.first_route;
  for (std::uint32_t visited = 0; visited < node.route_count && route != kInvalidIndex;
       ++visited) {
    if (route >= brain.route_capacity)
      return false;
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[route];
    const DirectRoute edge = brain.routes[route];
    if (meta.live != 0u && edge.implicit_family == family && edge.implicit_slot == virtual_slot)
      return true;
    route = edge.next_route;
  }
  return false;
}

__device__ inline DirectImplicitCandidateSet enumerate_direct_implicit_candidates(
    const DirectBrainV01& brain, std::uint32_t source, std::uint64_t context_signature,
    std::int32_t participation_threshold_q16 = kImplicitParticipationThresholdQ16) {
  DirectImplicitCandidateSet set{};
  set.count = 0u;
  if (source >= brain.node_count || brain.implicit.family_count == 0u)
    return set;
  const DirectNode node = brain.nodes[source];
  if (node.implicit_family == kInvalidIndex || node.implicit_family >= brain.implicit.family_count)
    return set;
  const DirectImplicitFamily family = brain.implicit.families[node.implicit_family];
  for (std::uint32_t local = 0; local < family.virtual_slot_count; ++local) {
    const std::uint32_t virtual_slot = family.first_virtual_slot + local;
    const std::uint32_t target = direct_implicit_target(family, source, virtual_slot);
    if (target == kInvalidIndex)
      continue;
    if (direct_implicit_is_materialized(brain, source, node.implicit_family, virtual_slot))
      continue;
    DirectImplicitException exception{};
    std::int32_t conductance =
        direct_implicit_conductance_q16(family, source, virtual_slot, context_signature);
    if (lookup_direct_implicit_exception(brain, node.implicit_family, source, virtual_slot,
                                         &exception)) {
      if ((exception.flags & kImplicitExceptionDisabled) != 0u)
        continue;
      conductance += exception.conductance_delta_q16;
    }
    if (conductance < participation_threshold_q16)
      continue;
    DirectImplicitCandidate cand{};
    cand.valid = 1u;
    cand.family = node.implicit_family;
    cand.virtual_slot = virtual_slot;
    cand.target = target;
    cand.delay = 1u + ((source + virtual_slot + family.lineage) & 3u);
    cand.conductance_q16 = conductance;
    if (set.count < kMaxImplicitActiveFanout) {
      std::uint32_t pos = set.count;
      while (pos > 0 &&
             (cand.conductance_q16 > set.candidates[pos - 1].conductance_q16 ||
              (cand.conductance_q16 == set.candidates[pos - 1].conductance_q16 &&
               cand.virtual_slot < set.candidates[pos - 1].virtual_slot))) {
        set.candidates[pos] = set.candidates[pos - 1];
        --pos;
      }
      set.candidates[pos] = cand;
      ++set.count;
    } else {
      const auto& worst = set.candidates[kMaxImplicitActiveFanout - 1];
      if (cand.conductance_q16 > worst.conductance_q16 ||
          (cand.conductance_q16 == worst.conductance_q16 && cand.virtual_slot < worst.virtual_slot)) {
        std::uint32_t pos = kMaxImplicitActiveFanout - 1;
        while (pos > 0 &&
               (cand.conductance_q16 > set.candidates[pos - 1].conductance_q16 ||
                (cand.conductance_q16 == set.candidates[pos - 1].conductance_q16 &&
                 cand.virtual_slot < set.candidates[pos - 1].virtual_slot))) {
          set.candidates[pos] = set.candidates[pos - 1];
          --pos;
        }
        set.candidates[pos] = cand;
      }
    }
  }
  return set;
}

__device__ inline DirectImplicitCandidate select_direct_implicit_candidate(
    const DirectBrainV01& brain, std::uint32_t source, std::uint64_t context_signature) {
  const DirectImplicitCandidateSet set =
      enumerate_direct_implicit_candidates(brain, source, context_signature, -2147483647);
  return set.count > 0 ? set.candidates[0] : DirectImplicitCandidate{};
}

__device__ inline DirectImplicitParticipationResult record_direct_implicit_participation(
    DirectBrainV01& brain, std::uint32_t family, std::uint32_t source,
    std::uint32_t virtual_slot, std::uint32_t tick, std::uint64_t external_root,
    bool is_external_origin, std::uint32_t min_distinct_external_roots = 2u,
    std::uint32_t min_participation_span = 0u) {
  DirectImplicitParticipationResult res{};
  const std::uint64_t key = direct_implicit_exception_key(family, source, virtual_slot);
  if (key == 0u || brain.implicit.exception_capacity == 0u)
    return res;
  std::uint32_t slot = direct_implicit_exception_bucket(key, brain.implicit.exception_capacity);
  for (std::uint32_t probe = 0; probe < 64u; ++probe) {
    DirectImplicitException& entry = brain.implicit.exceptions[slot];
    const std::uint64_t observed_key = atomicCAS(
        reinterpret_cast<unsigned long long*>(&entry.key), 0ULL, static_cast<unsigned long long>(key));
    if (observed_key == 0ULL) {
      atomicAdd(brain.implicit.exception_count, 1u);
    }
    if (observed_key == 0ULL || observed_key == key) {
      const std::uint32_t count = atomicAdd(&entry.participation_count, 1u) + 1u;
      res.participation_count = count;
      if (is_external_origin && external_root != 0u) {
        const unsigned long long observed_first = atomicCAS(
            reinterpret_cast<unsigned long long*>(&entry.first_external_root),
            0ULL, static_cast<unsigned long long>(external_root));
        if (observed_first != 0ULL && observed_first != external_root) {
          atomicExch(&entry.has_distinct_external_root, 1u);
        }
      }
      atomicCAS(&entry.first_participation_tick, 0u, tick);
      entry.last_participation_tick = tick;

      const bool has_first = entry.first_external_root != 0u;
      const bool has_distinct = entry.has_distinct_external_root != 0u;
      const std::uint32_t diversity = (has_first ? 1u : 0u) + (has_distinct ? 1u : 0u);
      res.external_diversity = diversity;
      const std::uint32_t first_tick = entry.first_participation_tick;
      const std::uint32_t current_flags = entry.flags;
      const bool meets_criteria =
          (diversity >= min_distinct_external_roots) &&
          ((tick >= first_tick) && (tick - first_tick >= min_participation_span)) &&
          ((current_flags & (kImplicitFlagMaterialized | kImplicitFlagPendingMaterialize)) == 0u);
      if (meets_criteria) {
        const std::uint32_t prev_flags = atomicOr(&entry.flags, kImplicitFlagPendingMaterialize);
        if ((prev_flags & (kImplicitFlagMaterialized | kImplicitFlagPendingMaterialize)) == 0u) {
          res.should_materialize = true;
        }
      }
      return res;
    }
    slot = (slot + 1u) & (brain.implicit.exception_capacity - 1u);
  }
  return res;
}
#endif

void initialize_direct_implicit_state(DirectBrainV01* brain,
                                      const DirectImplicitFamily* host_families,
                                      std::uint32_t family_count,
                                      std::uint64_t virtual_interaction_count,
                                      std::uint32_t exception_capacity);
void destroy_direct_implicit_state(DirectBrainV01* brain);

// Observer/test aperture for an exact sparse override. Resident learning does
// not require this call; normal lived specialization materializes an explicit
// route through #1198. The aperture is useful for focal lesion controls.
bool set_direct_implicit_exception(DirectBrainV01* brain, std::uint32_t family,
                                   std::uint32_t source, std::uint32_t virtual_slot,
                                   std::int32_t conductance_delta_q16,
                                   std::uint32_t flags);

#undef DIRECT_IMPLICIT_HD
#undef DIRECT_IMPLICIT_DEVICE

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_IMPLICIT_CAUSAL_MESH_CUH
