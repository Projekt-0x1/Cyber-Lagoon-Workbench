#ifndef HARDWARE_NATIVE_DIRECT_ADULT_BACKPRESSURE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_BACKPRESSURE_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_resource_ecology_abi.cuh"
#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_network_tube_chemistry.cuh"
#include "hardware_native/direct_adult_bounded_fanout.cuh"

namespace substrate::direct_adult {

// d.backpressure (#1520): inhibitory feedback propagates upstream through
// local contextual tubes when a downstream finite resource pool saturates,
// reducing UNVERIFIED speculative search without suppressing actual contact
// or manufacturing evidence.
//
// Law anchors (Revision 12 learning ecology SS9 anticipatory priming, SS14
// field/firewall, SS16 resource-backed chemistry; canonical architecture SS6
// authority floor):
//   * saturation is a measured finite-resource condition -- charged/capacity
//     of device-resident pools -- never a host-supplied semantic flag;
//   * every refusal writes a full identity receipt: downstream pool, feeder
//     route incarnation, delivery-path context signature, target node,
//     resident tick and finite expiry, folded into one nonzero identity.
//     Dampening is evidence with provenance, never a silent drop;
//   * the feedback rides the SAME channel physics as every other carried
//     signal: magnitude scales by route conductance and endpoint tube
//     chemistry, sign composes through the inhibitory sign law exactly as
//     signed_sparse_route_delivery_q16 composes it. An excitatory carrier
//     delivers inhibition upstream; an inhibitory carrier flips it;
//   * verification asymmetry keys on settled world_return LINEAGE -- an
//     exact verified world_return record on the charging context settles the
//     episode and exempts its prospective work. Pool-class membership alone
//     never decides; unverified records cannot self-confirm;
//   * dampening is a MONOTONE envelope of measured saturation, exposed for
//     successor-shadow / multi-horizon nomination to consume before
//     speculative work is spent, not a binary gate;
//   * relief and recovery are device-side: receipts expire against the
//     resident tick, drained pools resume service through the ecology ABI,
//     and no host counter write participates anywhere.
//
// Single-writer surface: callers serialize per frontier exactly like the
// other bounded resident frontiers; the plain stores below are order-stable
// under that discipline.

inline constexpr std::uint32_t kResidentBackpressureSaturationOneQ16 = 1u << 16;
inline constexpr std::uint32_t kResidentBackpressureReceiptCapacity = 8u;
// Pressure is finite: a receipt stops binding after this many resident ticks
// even if the downstream never drains, so recovery never depends on a reset.
inline constexpr std::uint32_t kResidentBackpressurePressureTicks = 8u;
// The delivery path folds route context as eligibility_context ^
// (source * 2654435761); backpressure identities use the same fold so a
// receipt names the exact tube instance the charge travelled on.
inline constexpr std::uint32_t kResidentBackpressureContextFold = 2654435761u;

__host__ __device__ inline std::uint32_t resident_pool_saturation_q16(
    const DirectResourcePoolState& pool) {
  if (pool.capacity_units == 0u) return 0u;
  const std::uint64_t charged =
      pool.charged_units > pool.capacity_units ? pool.capacity_units
                                               : pool.charged_units;
  return static_cast<std::uint32_t>(
      (charged << 16) / pool.capacity_units);
}

__host__ __device__ inline bool resident_pool_is_speculative(
    DirectResourcePoolKind kind) {
  return kind == DirectResourcePoolKind::topology_proposal ||
         kind == DirectResourcePoolKind::representation_source_state;
}

__host__ __device__ inline std::int32_t resident_backpressure_mul_q16(
    std::int32_t a, std::int32_t b) {
  return static_cast<std::int32_t>((static_cast<std::int64_t>(a) * b) >> 16);
}

// Monotone dampening envelope of measured saturation: exactly 0 at an empty
// pool, exactly One at/above the threshold, a pure function of the measured
// condition between. Nomination surfaces consume this before spending
// speculative work instead of branching on a binary gate.
__host__ __device__ inline std::uint32_t
resident_speculative_dampening_envelope_q16(std::uint32_t saturation_q16,
                                            std::uint32_t threshold_q16) {
  if (threshold_q16 == 0u || saturation_q16 >= threshold_q16)
    return kResidentBackpressureSaturationOneQ16;
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(saturation_q16) << 16) / threshold_q16);
}

// Verification asymmetry: only an exact verified world_return settled ON THIS
// CONTEXT settles the episode's lineage. Kind, verified bit and context must
// all agree; an unverified or endogenous record on the same context is the
// self-confirmation the firewall forbids, and a settled return elsewhere
// says nothing about this context.
__host__ __device__ inline bool resident_lineage_settled_world_return(
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint32_t context_signature) {
  if (records == nullptr) return false;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const direct_network::DirectExactHistoryRecord& record = records[i];
    if (record.kind != direct_network::DirectExactHistoryKind::world_return)
      continue;
    if ((record.flags & direct_network::kDirectHistoryVerifiedObservation) == 0u)
      continue;
    if (record.context != context_signature) continue;
    return true;
  }
  return false;
}

struct alignas(8) ResidentBackpressureReceipt {
  std::uint64_t identity;        // fold of every identity word below; nonzero
  std::uint64_t route_incarnation;
  std::uint32_t downstream_pool; // DirectResourcePoolKind of the saturated pool
  std::uint32_t target_node;     // downstream endpoint of the feeder route
  std::uint32_t context_signature;
  std::uint32_t resident_tick;
  std::uint32_t expiry_tick;  // resident_tick + kResidentBackpressurePressureTicks
  std::uint32_t pressure_q16;   // measured saturation at write time
  std::uint32_t dampening_q16;  // envelope sample at write time
  std::uint32_t live;
};
static_assert(std::is_trivial_v<ResidentBackpressureReceipt> &&
              std::is_standard_layout_v<ResidentBackpressureReceipt> &&
              std::has_unique_object_representations_v<ResidentBackpressureReceipt>);

struct ResidentBackpressureFrontier {
  ResidentBackpressureReceipt receipts[kResidentBackpressureReceiptCapacity];
  std::uint64_t receipts_written;
  std::uint64_t receipts_expired;
  // Feedback capacity exhausted: the refusal happened and is counted rather
  // than guessed, but no receipt could claim identity space.
  std::uint64_t refused_capacity;
};
static_assert(std::is_trivial_v<ResidentBackpressureFrontier> &&
              std::is_standard_layout_v<ResidentBackpressureFrontier> &&
              std::has_unique_object_representations_v<ResidentBackpressureFrontier>);

// Device-owned identity of the route whose charge meets the pressure.
struct ResidentBackpressureChargeContext {
  std::uint32_t context_signature;
  std::uint32_t target_node;
  std::uint64_t route_incarnation;
  std::uint32_t resident_tick;
  const direct_network::DirectExactHistoryRecord* history_records;
  std::uint32_t history_count;
};

struct ResidentBackpressureDecision {
  DirectResourcePoolKind downstream_kind;
  DirectResourcePoolKind speculative_kind;
  std::uint32_t saturation_q16;
  std::uint32_t threshold_q16;
  std::uint64_t deferred_units;
  bool dampened;  // full refusal: every requested unit deferred
  // Deepened fields: the measured envelope sample (before any lineage
  // exemption), whether settled world_return lineage exempted the charge,
  // and the identity of the receipt written for a partial or full refusal.
  std::uint32_t dampening_q16;
  std::uint32_t verified_lineage;
  std::uint64_t receipt_identity;
};

__host__ __device__ inline std::uint64_t resident_backpressure_receipt_identity(
    const ResidentBackpressureReceipt& receipt) {
  std::uint64_t identity = 0x62617072657373ull;
  identity = direct_network::exact_history_fold_word(identity, receipt.route_incarnation);
  identity = direct_network::exact_history_fold_word(identity, receipt.downstream_pool);
  identity = direct_network::exact_history_fold_word(identity, receipt.target_node);
  identity = direct_network::exact_history_fold_word(identity, receipt.context_signature);
  identity = direct_network::exact_history_fold_word(identity, receipt.resident_tick);
  return identity == 0u ? 1u : identity;
}

// Retires receipts whose finite horizon has passed. Device-side sweep driven
// by the resident tick argument; recovery never waits on a host reset.
__host__ __device__ inline void expire_resident_backpressure(
    ResidentBackpressureFrontier* frontier, std::uint32_t current_tick) {
  if (frontier == nullptr) return;
  for (std::uint32_t i = 0u; i < kResidentBackpressureReceiptCapacity; ++i) {
    ResidentBackpressureReceipt& receipt = frontier->receipts[i];
    if (receipt.live == 0u || current_tick < receipt.expiry_tick) continue;
    receipt.live = 0u;
    ++frontier->receipts_expired;
  }
}

__host__ __device__ inline std::uint64_t write_resident_backpressure_receipt(
    ResidentBackpressureFrontier* frontier, DirectResourcePoolKind downstream_kind,
    const ResidentBackpressureChargeContext& charge, std::uint32_t saturation_q16,
    std::uint32_t dampening_q16) {
  for (std::uint32_t i = 0u; i < kResidentBackpressureReceiptCapacity; ++i) {
    ResidentBackpressureReceipt& receipt = frontier->receipts[i];
    if (receipt.live != 0u) continue;
    receipt.identity = 0u;
    receipt.route_incarnation = charge.route_incarnation;
    receipt.downstream_pool = static_cast<std::uint32_t>(downstream_kind);
    receipt.target_node = charge.target_node;
    receipt.context_signature = charge.context_signature;
    receipt.resident_tick = charge.resident_tick;
    receipt.expiry_tick = charge.resident_tick + kResidentBackpressurePressureTicks;
    receipt.pressure_q16 = saturation_q16;
    receipt.dampening_q16 = dampening_q16;
    receipt.live = 1u;
    receipt.identity = resident_backpressure_receipt_identity(receipt);
    ++frontier->receipts_written;
    return receipt.identity;
  }
  ++frontier->refused_capacity;
  return 0u;
}

// Charges `requested_units` of `speculative_kind` along the charging route
// named by `charge`, unless measured downstream saturation dampens it: the
// monotone envelope defers a proportional share of the request (booked
// deferred AND rejected on both pools -- evidence, never a silent drop) and
// writes one identity-complete receipt per refusal event. An episode whose
// lineage carries a settled verified world_return on the charging context is
// exempt: verified prospective work passes untouched by another pool's
// pressure, while ordinary speculative-pool capacity still binds it. Returns
// false on malformed arguments only; any dampening is a true return.
__host__ __device__ inline bool apply_resident_backpressure(
    DirectResourceEcologyState* ecology, ResidentBackpressureFrontier* frontier,
    DirectResourcePoolKind downstream_kind, DirectResourcePoolKind speculative_kind,
    std::uint64_t requested_units, std::uint32_t threshold_q16,
    const ResidentBackpressureChargeContext& charge, ResidentBackpressureDecision* out) {
  if (out == nullptr || ecology == nullptr || requested_units == 0u ||
      threshold_q16 == 0u ||
      threshold_q16 > kResidentBackpressureSaturationOneQ16 ||
      !resident_pool_is_speculative(speculative_kind) ||
      downstream_kind == speculative_kind ||
      static_cast<std::uint32_t>(downstream_kind) >=
          static_cast<std::uint32_t>(DirectResourcePoolKind::count))
    return false;
  DirectResourcePoolState& downstream =
      ecology->pools[static_cast<std::uint32_t>(downstream_kind)];
  DirectResourcePoolState& speculative =
      ecology->pools[static_cast<std::uint32_t>(speculative_kind)];
  out->downstream_kind = downstream_kind;
  out->speculative_kind = speculative_kind;
  out->threshold_q16 = threshold_q16;
  out->saturation_q16 = resident_pool_saturation_q16(downstream);
  out->dampening_q16 =
      resident_speculative_dampening_envelope_q16(out->saturation_q16, threshold_q16);
  out->verified_lineage =
      resident_lineage_settled_world_return(charge.history_records,
                                            charge.history_count,
                                            charge.context_signature)
          ? 1u
          : 0u;
  out->deferred_units = 0u;
  out->receipt_identity = 0u;
  const std::uint64_t share = out->verified_lineage != 0u
                                  ? 0u
                                  : (requested_units * out->dampening_q16) >> 16;
  if (share > 0u) {
    downstream.deferred_units += share;
    speculative.deferred_units += share;
    speculative.rejected_units += share;
    out->deferred_units = share;
    if (frontier != nullptr)
      out->receipt_identity = write_resident_backpressure_receipt(
          frontier, downstream_kind, charge, out->saturation_q16,
          out->dampening_q16);
  }
  out->dampened = out->deferred_units == requested_units;
  const std::uint64_t remaining = requested_units - out->deferred_units;
  if (remaining == 0u) return true;
  const std::uint64_t bytes =
      remaining *
      (speculative.bytes_per_unit != 0u ? speculative.bytes_per_unit : 1u);
  if (speculative.charged_units + remaining > speculative.capacity_units ||
      ecology->global_charged_bytes + bytes > ecology->global_capacity_bytes) {
    speculative.rejected_units += remaining;  // ordinary capacity refusal
    return true;
  }
  speculative.charged_units += remaining;
  speculative.live_units += remaining;
  if (speculative.charged_units > speculative.high_water_units)
    speculative.high_water_units = speculative.charged_units;
  ecology->global_charged_bytes += bytes;
  if (ecology->global_charged_bytes > ecology->global_high_water_bytes)
    ecology->global_high_water_bytes = ecology->global_charged_bytes;
  return true;
}

struct ResidentBackpressurePropagation {
  std::int32_t delivered_q16;   // summed signed feedback through the carriers
  std::uint32_t magnitude_q16;  // summed |delivery|, saturating at One
  std::uint32_t matched_receipts;
  std::uint64_t refused_inactive_routes;
  std::uint64_t refused_wrong_context;
  std::uint64_t refused_stale_incarnation;
};

// Upstream propagation over the strictly bounded active-route neighborhood of
// the feeder source: every explicitly deactivated route inside the declared
// window refuses, a live receipt on the right target under the wrong tube
// context refuses, a live receipt behind a stale route incarnation refuses,
// and only the exact current tube instance delivers -- its measured pressure
// carried negative through the shared delivery composition (route
// conductance x endpoint tube chemistry x inhibitory sign law), so the
// consuming source reads a signed, chemistry-modulated dampening magnitude.
__host__ __device__ inline ResidentBackpressurePropagation
propagate_resident_backpressure(
    const direct_network::DirectNode* nodes, std::uint32_t node_count,
    const direct_network::DirectRoute* routes, std::uint32_t route_capacity,
    const std::uint64_t* route_incarnations, std::uint32_t source_node,
    const ResidentBackpressureFrontier* frontier, std::uint32_t current_tick,
    std::int32_t competition_gain_q16, bool honor_inhibitory_sign) {
  ResidentBackpressurePropagation out{};
  if (nodes == nullptr || routes == nullptr || route_incarnations == nullptr ||
      frontier == nullptr || source_node >= node_count)
    return out;
  const direct_network::DirectNode& source = nodes[source_node];
  const std::uint32_t window = direct_adult_core::bounded_active_route_count(source);
  for (std::uint32_t k = 0u; k < window; ++k) {
    const std::uint32_t slot = source.route_offset + k;
    if (slot >= route_capacity) break;
    const direct_network::DirectRoute& route = routes[slot];
    if ((route.flags & direct_network::kRouteFlagActive) == 0u) {
      ++out.refused_inactive_routes;
      continue;
    }
    if (route.target >= node_count) continue;
    const std::uint32_t context_signature =
        route.eligibility_context ^
        (source_node * kResidentBackpressureContextFold);
    for (std::uint32_t i = 0u; i < kResidentBackpressureReceiptCapacity; ++i) {
      const ResidentBackpressureReceipt& receipt = frontier->receipts[i];
      if (receipt.live == 0u || current_tick >= receipt.expiry_tick) continue;
      if (receipt.target_node != route.target) continue;  // unrelated downstream
      if (receipt.context_signature != context_signature) {
        ++out.refused_wrong_context;
        continue;
      }
      if (route_incarnations[slot] != receipt.route_incarnation) {
        ++out.refused_stale_incarnation;
        continue;
      }
      std::int32_t delivered = resident_backpressure_mul_q16(
          route.conductance_q16,
          -static_cast<std::int32_t>(receipt.pressure_q16));
      delivered = resident_backpressure_mul_q16(
          delivered,
          direct_network::direct_tube_chemistry_q16(source.chemotype,
                                                    nodes[route.target].chemotype)
              .conductance_gain_q16);
      const bool inhibitory =
          (route.flags & direct_network::kRouteFlagInhibitory) != 0u ||
          (source.flags & direct_network::kNodeFlagInhibitory) != 0u;
      if (honor_inhibitory_sign && inhibitory)
        delivered =
            -resident_backpressure_mul_q16(delivered, competition_gain_q16);
      out.delivered_q16 = static_cast<std::int32_t>(
          static_cast<std::int64_t>(out.delivered_q16) + delivered);
      const std::int32_t magnitude = delivered < 0 ? -delivered : delivered;
      const std::int32_t room = static_cast<std::int32_t>(
          kResidentBackpressureSaturationOneQ16 - out.magnitude_q16);
      out.magnitude_q16 +=
          static_cast<std::uint32_t>(magnitude < room ? magnitude : room);
      ++out.matched_receipts;
      break;
    }
  }
  return out;
}

// Downstream consumption for successor-shadow / multi-horizon nomination:
// nominated speculative work shrinks monotonically as propagated pressure
// rises, reaching zero only at full measured One.
__host__ __device__ inline std::uint32_t resident_speculative_work_after_backpressure(
    std::uint32_t nominated_units, std::uint32_t dampening_magnitude_q16) {
  const std::uint32_t dampening =
      dampening_magnitude_q16 > kResidentBackpressureSaturationOneQ16
          ? kResidentBackpressureSaturationOneQ16
          : dampening_magnitude_q16;
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(nominated_units) *
       (kResidentBackpressureSaturationOneQ16 - dampening)) >>
      16);
}

}  // namespace substrate::direct_adult

#endif
