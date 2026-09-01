#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MULTIMODAL_GROUNDING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MULTIMODAL_GROUNDING_CUH

// f.multimodal_grounding (#1515): cross-modal associative binding linking
// auditory, visual and tactile contact into unified objects.
//
// Law anchors (Revision 12):
//   * a unified object is an observer description of recurring cross-modal
//     organization -- this header grows exactly such organization from
//     actual membrane-admitted contact and owns no semantic object class;
//   * binding follows the actual chronology of multi-channel ingress: two
//     contacts contribute evidence only when they were admitted on distinct
//     channels within one window of device-resident ticks. Identical payload
//     bytes arranged without temporal coherence refuse to bind;
//   * channel bytes stay opaque at admission: deposits read channel, value
//     and resident tick from device exact history and never consult any
//     host-supplied tag, label or context field;
//   * participation is never fabricated: only contacts carrying a real
//     sensory_contact record contribute, and consequence closure attaches
//     only where a settled world return follows real multi-channel
//     participation around the originating action's emission.

#include <cstdint>

#include "hardware_native/direct_adult_multimodal_grounding_abi.cuh"
#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

// Keep Q16 constants qualified so this header does not export core names.
using direct_adult_core::mul_q16;

__device__ inline std::int32_t grounding_rise_q16(std::int32_t mass) {
  const std::int32_t headroom = direct_adult_core::kQ16One - mass;
  return mass + direct_adult_core::mul_q16(headroom > 0 ? headroom : 0,
                                           kGroundingRiseGainQ16);
}

// Recurring organization: one coincident window opens no binding. Mass
// crosses the bound threshold only after repeated coherent co-occurrence.
__device__ inline bool grounding_pair_strong(const DirectCrossModalPair& pair) {
  return pair.bind_mass_q16 >= kGroundingBoundThresholdQ16;
}

__device__ inline void grounding_note_exposure(
    DirectMultimodalGroundingTable* table, std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < table->exposure_count; ++i) {
    if (table->exposures[i].channel == channel) {
      ++table->exposures[i].contacts;
      return;
    }
  }
  if (table->exposure_count >= kGroundingExposureCapacity) {
    atomicAdd(&table->fence_refusals, 1u);
    return;
  }
  table->exposures[table->exposure_count++] =
      DirectChannelExposure{channel, 1u};
}

__device__ inline std::int32_t grounding_find_pair(
    const DirectMultimodalGroundingTable* table, std::uint32_t channel_a,
    std::uint32_t value_a, std::uint32_t channel_b, std::uint32_t value_b) {
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectCrossModalPair& pair = table->pairs[i];
    if (pair.channel_a == channel_a && pair.value_a == value_a &&
        pair.channel_b == channel_b && pair.value_b == value_b)
      return static_cast<std::int32_t>(i);
  }
  return -1;
}

// Canonical-form lookup: the table stores each pair channel-sorted, so a
// caller holding the constituents in temporal order (surface heard after the
// ground, or either order of a coordinated compound) normalizes before
// matching. Ascending queries are unchanged.
__device__ inline std::int32_t grounding_find_pair_canonical(
    const DirectMultimodalGroundingTable* table, std::uint32_t channel_a,
    std::uint32_t value_a, std::uint32_t channel_b, std::uint32_t value_b) {
  if (channel_a > channel_b ||
      (channel_a == channel_b && value_a > value_b)) {
    const std::uint32_t channel = channel_a;
    const std::uint32_t value = value_a;
    channel_a = channel_b;
    value_a = value_b;
    channel_b = channel;
    value_b = value;
  }
  return grounding_find_pair(table, channel_a, value_a, channel_b, value_b);
}

__device__ inline void grounding_deposit_pair(
    DirectMultimodalGroundingTable* table, const DirectCrossModalContact& a,
    const DirectCrossModalContact& b) {
  if (a.channel == b.channel) return;
  const bool a_first = a.channel < b.channel ||
                       (a.channel == b.channel && a.value <= b.value);
  const DirectCrossModalContact& lo = a_first ? a : b;
  const DirectCrossModalContact& hi = a_first ? b : a;
  const std::int32_t slot =
      grounding_find_pair(table, lo.channel, lo.value, hi.channel, hi.value);
  if (slot >= 0) {
    DirectCrossModalPair& pair = table->pairs[slot];
    ++pair.cooccurrences;
    pair.bind_mass_q16 = grounding_rise_q16(pair.bind_mass_q16);
    pair.last_tick = hi.resident_tick;
    return;
  }
  if (table->pair_count >= kGroundingPairCapacity) {
    atomicAdd(&table->fence_refusals, 1u);
    return;
  }
  DirectCrossModalPair fresh{};
  fresh.channel_a = lo.channel;
  fresh.value_a = lo.value;
  fresh.channel_b = hi.channel;
  fresh.value_b = hi.value;
  fresh.cooccurrences = 1u;
  fresh.bind_mass_q16 = grounding_rise_q16(0);
  fresh.first_tick = lo.resident_tick;
  fresh.last_tick = hi.resident_tick;
  table->pairs[table->pair_count++] = fresh;
}

// One chronological pass over device exact history. Every admitted
// sensory contact is held against prior contacts still inside the coherence
// window; distinct-channel neighbours deposit one cross-modal evidence unit.
// Payload bytes are compared for equality only -- no host field participates.
__device__ inline void grounding_ingest_history(
    DirectMultimodalGroundingTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  DirectCrossModalContact recent[kGroundingRecentCapacity];
  std::uint32_t recent_count = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    const DirectCrossModalContact contact{record.subject, record.value,
                                          record.resident_tick};
    grounding_note_exposure(table, contact.channel);
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick -
                                          recent[j].resident_tick
                                    : 0u;
      if (age > kGroundingCoherenceWindowTicks) continue;
      grounding_deposit_pair(table, recent[j], contact);
    }
    std::uint32_t kept = 0u;
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick -
                                          recent[j].resident_tick
                                    : 0u;
      if (age <= kGroundingCoherenceWindowTicks &&
          kept < kGroundingRecentCapacity)
        recent[kept++] = recent[j];
    }
    if (kept < kGroundingRecentCapacity) recent[kept++] = contact;
    recent_count = kept;
  }
}

// Unified objects emerge here: strongly bound pairs are joined when they
// share an exact leg, and each connected closure becomes one grounded object
// whose weakest member pair bounds its closure mass. Deterministic order:
// objects follow their smallest member-pair index, legs sort by
// (channel, value).
__device__ inline void grounding_absorb_pair(
    DirectGroundedObject& object, const DirectCrossModalPair& pair) {
  const DirectCrossModalContact legs[2] = {
      {pair.channel_a, pair.value_a, pair.first_tick},
      {pair.channel_b, pair.value_b, pair.last_tick}};
  for (std::uint32_t m = 0u; m < 2u; ++m) {
    bool present = false;
    for (std::uint32_t l = 0u; l < object.leg_count && !present; ++l)
      present = object.legs[l].channel == legs[m].channel &&
                object.legs[l].value == legs[m].value;
    if (present || object.leg_count >= kGroundingLegCapacity) continue;
    std::uint32_t insert = object.leg_count;
    while (insert > 0u &&
           (object.legs[insert - 1u].channel > legs[m].channel ||
            (object.legs[insert - 1u].channel == legs[m].channel &&
             object.legs[insert - 1u].value > legs[m].value))) {
      object.legs[insert] = object.legs[insert - 1u];
      --insert;
    }
    object.legs[insert] = legs[m];
    ++object.leg_count;
  }
  if (pair.bind_mass_q16 < object.closure_mass_q16)
    object.closure_mass_q16 = pair.bind_mass_q16;
  if (pair.first_tick < object.first_tick) object.first_tick = pair.first_tick;
  if (pair.last_tick > object.last_tick) object.last_tick = pair.last_tick;
}

// Unified objects emerge here: strongly bound pairs are joined transitively
// whenever they share an exact leg, and each connected closure becomes one
// grounded object whose weakest member pair bounds its closure mass.
// Deterministic order: objects follow their smallest member-pair index, legs
// sort by (channel, value).
__device__ inline void grounding_extract_objects(
    DirectMultimodalGroundingTable* table) {
  table->object_count = 0u;
  bool merged[kGroundingPairCapacity] = {};
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    if (!grounding_pair_strong(table->pairs[i]) || merged[i]) continue;
    DirectGroundedObject object{};
    object.closure_mass_q16 = direct_adult_core::kQ16One;
    object.first_tick = 0xffffffffu;
    merged[i] = true;
    grounding_absorb_pair(object, table->pairs[i]);
    bool grew = true;
    while (grew) {
      grew = false;
      for (std::uint32_t j = 0u; j < table->pair_count; ++j) {
        if (!grounding_pair_strong(table->pairs[j]) || merged[j]) continue;
        const DirectCrossModalPair& pair = table->pairs[j];
        bool joins = false;
        for (std::uint32_t l = 0u; l < object.leg_count && !joins; ++l)
          joins = (pair.channel_a == object.legs[l].channel &&
                   pair.value_a == object.legs[l].value) ||
                  (pair.channel_b == object.legs[l].channel &&
                   pair.value_b == object.legs[l].value);
        if (!joins) continue;
        merged[j] = true;
        grounding_absorb_pair(object, pair);
        grew = true;
      }
    }
    if (table->object_count >= kGroundingObjectCapacity) {
      atomicAdd(&table->fence_refusals, 1u);
      return;
    }
    table->objects[table->object_count++] = object;
  }
}

// Partial-modal integrity: dissolving one channel removes exactly the
// bindings that carry it and touches nothing else. Surviving pairs keep
// their learned bytes; unified organizations reform from the survivors.
__device__ inline std::uint32_t grounding_dissolve_channel(
    DirectMultimodalGroundingTable* table, std::uint32_t channel) {
  std::uint32_t kept = 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectCrossModalPair& pair = table->pairs[i];
    if (pair.channel_a == channel || pair.channel_b == channel) {
      ++removed;
      continue;
    }
    table->pairs[kept++] = pair;
  }
  table->pair_count = kept;
  grounding_extract_objects(table);
  return removed;
}

__device__ inline std::int32_t grounding_object_with_leg(
    const DirectMultimodalGroundingTable* table, std::uint32_t channel,
    std::uint32_t value) {
  for (std::uint32_t i = 0u; i < table->object_count; ++i) {
    const DirectGroundedObject& object = table->objects[i];
    for (std::uint32_t l = 0u; l < object.leg_count; ++l)
      if (object.legs[l].channel == channel && object.legs[l].value == value)
        return static_cast<std::int32_t>(i);
  }
  return -1;
}

// Consequence closure over the sensorimotor loop (#1507): a settled world
// return strengthens exactly the grounded organizations whose legs actually
// participated within the consequence window before its originating action's
// emission. Returns with no participating legs credit nothing.
__device__ inline bool grounding_credit_world_return(
    DirectMultimodalGroundingTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord& world_return) {
  bool found_emission = false;
  std::uint32_t emission_tick = 0u;
  for (std::uint32_t i = 0u; i < count && !found_emission; ++i) {
    if (records[i].kind == DirectExactHistoryKind::motor_output &&
        records[i].identity == world_return.identity) {
      emission_tick = records[i].resident_tick;
      found_emission = true;
    }
  }
  if (!found_emission) {
    ++table->consequences_unbound;
    return false;
  }
  bool credited = false;
  std::int32_t last_object = -1;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    if (record.resident_tick > emission_tick) continue;
    if (emission_tick - record.resident_tick >
        kGroundingConsequenceWindowTicks)
      continue;
    const std::int32_t object_slot = grounding_object_with_leg(
        table, record.subject, record.value);
    if (object_slot < 0 || object_slot == last_object) continue;
    ++table->objects[object_slot].consequence_samples;
    last_object = object_slot;
    credited = true;
  }
  if (!credited) ++table->consequences_unbound;
  return credited;
}

// Participation fence for consequences: only a world_return actually recorded
// in device exact history may distribute closure credit. A fabricated
// identity is refused and mints nothing.
__device__ inline bool grounding_credit_consequence_identity(
    DirectMultimodalGroundingTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t identity) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind == DirectExactHistoryKind::world_return &&
        records[i].identity == identity)
      return grounding_credit_world_return(table, records, count, records[i]);
  }
  atomicAdd(&table->fence_refusals, 1u);
  return false;
}

__device__ inline void grounding_bind_all_consequences(
    DirectMultimodalGroundingTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::world_return) continue;
    ++table->consequences_processed;
    grounding_credit_world_return(table, records, count, records[i]);
  }
}

}  // namespace substrate::direct_network

#endif
