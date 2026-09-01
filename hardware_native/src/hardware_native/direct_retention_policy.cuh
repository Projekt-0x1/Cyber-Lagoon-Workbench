#ifndef HARDWARE_NATIVE_DIRECT_RETENTION_POLICY_CUH
#define HARDWARE_NATIVE_DIRECT_RETENTION_POLICY_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_adult {

#if defined(__CUDACC__)
#define DIRECT_RETENTION_HD __host__ __device__
#define DIRECT_RETENTION_DEVICE __device__
#else
#define DIRECT_RETENTION_HD
#define DIRECT_RETENTION_DEVICE
#endif

// Decisive evaluation of candidate rank for reclaim under rich policy.
// Returns true if candidate 'a' should be reclaimed BEFORE candidate 'b'
// (i.e. 'a' has higher reclaim priority / is more expendable).
DIRECT_RETENTION_HD inline bool compare_reclaim_candidates_rich(
    const DirectRetentionState& a, const DirectRetentionState& b, std::uint64_t current_tick) {
  // 1. Hard pins: pinned candidates must NOT be reclaimed before unpinned ones.
  const bool a_pinned = (a.flags & kDirectRetentionHardPinned) != 0u || a.pin_reasons != 0u;
  const bool b_pinned = (b.flags & kDirectRetentionHardPinned) != 0u || b.pin_reasons != 0u;
  if (a_pinned != b_pinned) {
    return !a_pinned;  // unpinned comes first (more expendable)
  }

  // 2. Damage + support: damaged but supported objects should be repaired, not reclaimed.
  const bool a_damaged_supported = ((a.flags & kDirectRetentionDamaged) != 0u) && (a.support_ema_q16 > 0);
  const bool b_damaged_supported = ((b.flags & kDirectRetentionDamaged) != 0u) && (b.support_ema_q16 > 0);
  if (a_damaged_supported != b_damaged_supported) {
    return !a_damaged_supported;
  }

  // 3. Quiet protection: protected items survive silence.
  const bool a_protected = (a.quiet_protect_until > current_tick) ||
                           ((a.flags & kDirectRetentionQuietProtected) != 0u);
  const bool b_protected = (b.quiet_protect_until > current_tick) ||
                           ((b.flags & kDirectRetentionQuietProtected) != 0u);
  if (a_protected != b_protected) {
    return !a_protected;  // unprotected is more expendable
  }

  // 4. Behavioral consequence: lower causal support is more expendable.
  if (a.support_ema_q16 != b.support_ema_q16) {
    return a.support_ema_q16 < b.support_ema_q16;
  }

  // 5. Prediction surprise: lower actual contradiction is more expendable.
  // Surprise is evidence to retain, not a prediction error to punish.
  if (a.contradiction_ema_q16 != b.contradiction_ema_q16) {
    return a.contradiction_ema_q16 < b.contradiction_ema_q16;
  }

  // 6. Usage: lower usage is more expendable.
  if (a.usage_ema_q16 != b.usage_ema_q16) {
    return a.usage_ema_q16 < b.usage_ema_q16;
  }

  // 7. Last use tick: older last use is more expendable.
  if (a.last_use_tick != b.last_use_tick) {
    return a.last_use_tick < b.last_use_tick;
  }

  // 8. Deterministic stable tie-breaks:
  if (a.logical_source != b.logical_source) {
    return a.logical_source < b.logical_source;
  }
  if (a.logical_slot != b.logical_slot) {
    return a.logical_slot < b.logical_slot;
  }
  return a.logical_generation < b.logical_generation;
}

// Minimal policy reclaim comparison (#1200):
// Uses only local activity/mismatch without causal support state.
DIRECT_RETENTION_HD inline bool compare_reclaim_candidates_minimal(
    const DirectMinimalRetentionState& a, const DirectMinimalRetentionState& b,
    std::uint32_t a_source, std::uint32_t a_slot,
    std::uint32_t b_source, std::uint32_t b_slot,
    std::uint64_t current_tick) {
  (void)current_tick;
  // 1. Mismatch count: higher mismatch is more expendable.
  if (a.mismatch_count != b.mismatch_count) {
    return a.mismatch_count > b.mismatch_count;
  }

  // 2. Activity: lower activity is more expendable.
  if (a.activity_ema_q16 != b.activity_ema_q16) {
    return a.activity_ema_q16 < b.activity_ema_q16;
  }

  // 3. Last use tick: older is more expendable.
  if (a.last_use_tick != b.last_use_tick) {
    return a.last_use_tick < b.last_use_tick;
  }

  // 4. Stable tie-break:
  if (a_source != b_source) {
    return a_source < b_source;
  }
  return a_slot < b_slot;
}

// Decisive evaluation of repair rank for damaged candidates.
// Returns true if candidate 'a' should be repaired BEFORE candidate 'b'.
DIRECT_RETENTION_HD inline bool compare_repair_candidates(
    const DirectRetentionState& a, const DirectRetentionState& b) {
  const bool a_damaged = (a.flags & kDirectRetentionDamaged) != 0u;
  const bool b_damaged = (b.flags & kDirectRetentionDamaged) != 0u;
  if (a_damaged != b_damaged) {
    return a_damaged;
  }

  // Higher support has higher repair priority.
  if (a.support_ema_q16 != b.support_ema_q16) {
    return a.support_ema_q16 > b.support_ema_q16;
  }

  // Higher repair evidence has higher priority.
  if (a.repair_evidence_q16 != b.repair_evidence_q16) {
    return a.repair_evidence_q16 > b.repair_evidence_q16;
  }

  if (a.logical_source != b.logical_source) {
    return a.logical_source < b.logical_source;
  }
  if (a.logical_slot != b.logical_slot) {
    return a.logical_slot < b.logical_slot;
  }
  return a.logical_generation < b.logical_generation;
}

#if defined(__CUDACC__)
DIRECT_RETENTION_DEVICE inline void device_record_retention_use(
    DirectRetentionState& retention, std::uint64_t tick, std::int32_t conductance_q16) {
  retention.last_use_tick = tick;
  retention.last_confirmed_conductance_q16 = conductance_q16;
  // EMA update: alpha = 1/8
  retention.usage_ema_q16 = (retention.usage_ema_q16 * 7u + (1u << 16)) / 8u;
}

DIRECT_RETENTION_DEVICE inline void device_record_retention_support(
    DirectRetentionState& retention, std::int32_t credit_q16, std::uint64_t tick) {
  retention.last_support_tick = tick;
  if (credit_q16 > 0) {
    retention.support_ema_q16 = static_cast<std::int32_t>(
        (static_cast<std::int64_t>(retention.support_ema_q16) * 7ll +
         static_cast<std::int64_t>(credit_q16)) / 8ll);
  } else if (credit_q16 < 0) {
    const std::uint32_t contradiction = static_cast<std::uint32_t>(-credit_q16);
    retention.contradiction_ema_q16 =
        (retention.contradiction_ema_q16 * 7u + contradiction) / 8u;
    retention.support_ema_q16 = static_cast<std::int32_t>(
        (static_cast<std::int64_t>(retention.support_ema_q16) * 7ll +
         static_cast<std::int64_t>(credit_q16)) / 8ll);
  }
}

// Record only an already-settled physical consequence. The caller owns the
// exact-history transaction and generation-current participation check; this
// helper preserves their two independent measurements instead of collapsing
// them into a reward or semantic importance scalar.
DIRECT_RETENTION_DEVICE inline void device_record_causal_difference(
    DirectRetentionState& retention, std::uint32_t logical_source,
    std::uint32_t logical_slot, std::uint64_t logical_generation,
    std::uint32_t mismatch_bits, std::uint32_t consequence_magnitude_q16,
    std::uint64_t tick, std::int32_t conductance_q16) {
  if (retention.logical_source != logical_source ||
      retention.logical_slot != logical_slot ||
      retention.logical_generation != logical_generation) {
    retention = DirectRetentionState{};
    retention.logical_source = logical_source;
    retention.logical_slot = logical_slot;
    retention.logical_generation = logical_generation;
    retention.source_revision = logical_generation;
  }
  const std::uint32_t bounded_mismatch = mismatch_bits > 32u ? 32u : mismatch_bits;
  const std::uint32_t surprise_q16 = bounded_mismatch * ((1u << 16) / 32u);
  retention.contradiction_ema_q16 =
      (retention.contradiction_ema_q16 * 7u + surprise_q16) / 8u;
  retention.support_ema_q16 = static_cast<std::int32_t>(
      (static_cast<std::int64_t>(retention.support_ema_q16) * 7ll +
       consequence_magnitude_q16) / 8ll);
  retention.last_support_tick = tick;
  device_record_retention_use(retention, tick, conductance_q16);
}

DIRECT_RETENTION_DEVICE inline void device_record_retention_damage(
    DirectRetentionState& retention) {
  retention.flags |= kDirectRetentionDamaged;
}

DIRECT_RETENTION_DEVICE inline void device_record_retention_repair(
    DirectRetentionState& retention, std::uint32_t repair_evidence) {
  retention.flags &= ~kDirectRetentionDamaged;
  retention.flags |= kDirectRetentionRepaired;
  retention.repair_evidence_q16 += repair_evidence;
}
#endif

void initialize_direct_retention_bank(
    DirectRetentionState* retention_bank, std::uint32_t route_capacity, cudaStream_t stream = nullptr);

void initialize_direct_minimal_retention_bank(
    DirectMinimalRetentionState* minimal_bank, std::uint32_t route_capacity, cudaStream_t stream = nullptr);

#undef DIRECT_RETENTION_HD
#undef DIRECT_RETENTION_DEVICE

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_RETENTION_POLICY_CUH
