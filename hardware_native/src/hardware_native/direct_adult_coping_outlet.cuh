#ifndef HARDWARE_NATIVE_DIRECT_ADULT_COPING_OUTLET_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_COPING_OUTLET_CUH

// d.coping_outlet_modulation (#1535): stress attenuation through access to
// alternative physical/social action outlets blunting allostatic load.
//
// Law anchors (Revision 12 learning ecology §3-§5, development atlas NET15/
// NET17, canonical architecture §6-§7 no-authority-laundering):
//   * an alternative action OUTLET is a device-owned channel identity; USING
//     it means an actually-dispatched motor action on that channel SETTLED
//     through the closed consequence loop -- an exact verified world_return
//     record carrying the positive settlement resource delta. Transport
//     acceptance without settlement counts nothing;
//   * blunting acts only on the chronic load integrator of the allostatic
//     ledger: each sustained contact's load sample is attenuated monotonically
//     by settled outlet usage inside a bounded recency window, pinned by a
//     ceiling so coping can never invert into amplification. The acute path
//     and the drifted parameters are written by the parent walk alone;
//   * specificity: outlet existence attenuates nothing; settled returns on
//     unregistered channels contribute zero usage because attribution keys on
//     device-owned channel identity alone;
//   * parity: modalities differ only by channel identity; each independently
//     attenuates and contributions compose under the shared cap;
//   * no laundering: attenuation never mints participation, eligibility, or
//     credit and never edits exact history, tickets, or metrics.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_allostatic_metaplasticity.cuh"
#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kCopingOutletCapacity = 8u;
// Settled outlet usage blunts only inside this many resident ticks, mirroring
// the allostatic window scale: stale relief is not relief.
inline constexpr std::uint32_t kCopingUsageWindowTicks = 8u;
// Per settled action in window, and the hard ceiling on total attenuation.
inline constexpr std::int32_t kCopingPerActionBluntQ16 =
    direct_adult_core::kQ16One / 8;
inline constexpr std::int32_t kCopingBluntCeilingQ16 =
    direct_adult_core::kQ16One / 2;

struct DirectCopingOutlet {
  std::uint32_t channel;
  std::uint32_t settled_actions;
  std::uint32_t last_settled_tick;
};
static_assert(std::is_trivial_v<DirectCopingOutlet> &&
              std::is_standard_layout_v<DirectCopingOutlet> &&
              std::has_unique_object_representations_v<DirectCopingOutlet>);

struct DirectCopingLedger {
  DirectCopingOutlet outlets[kCopingOutletCapacity];
  std::uint32_t count;
  // Verified world returns on channels nobody registered as an outlet: real
  // settlements, counted, and attributed to no usage.
  std::uint32_t unregistered_returns;
  // Dispatched outlet-channel actions whose loop never closed: observed, and
  // never counted as usage.
  std::uint32_t unsettled_dispatches;
};
static_assert(std::is_trivial_v<DirectCopingLedger> &&
              std::is_standard_layout_v<DirectCopingLedger> &&
              std::has_unique_object_representations_v<DirectCopingLedger>);

__host__ __device__ inline DirectCopingOutlet* coping_find_outlet(
    DirectCopingLedger* ledger, std::uint32_t channel) {
  if (ledger == nullptr) return nullptr;
  for (std::uint32_t i = 0u; i < ledger->count; ++i)
    if (ledger->outlets[i].channel == channel) return &ledger->outlets[i];
  return nullptr;
}

// Registration grants existence only. It deposits no usage and attenuates
// nothing until a settled world return lands on the channel.
__host__ __device__ inline DirectCopingOutlet* coping_register_outlet(
    DirectCopingLedger* ledger, std::uint32_t channel) {
  if (ledger == nullptr) return nullptr;
  DirectCopingOutlet* found = coping_find_outlet(ledger, channel);
  if (found != nullptr) return found;
  if (ledger->count >= kCopingOutletCapacity) return nullptr;
  DirectCopingOutlet fresh{};
  fresh.channel = channel;
  ledger->outlets[ledger->count] = fresh;
  return &ledger->outlets[ledger->count++];
}

__host__ __device__ inline bool coping_outlet_registered(
    const DirectCopingLedger* ledger, std::uint32_t channel) {
  if (ledger == nullptr) return false;
  for (std::uint32_t i = 0u; i < ledger->count; ++i)
    if (ledger->outlets[i].channel == channel) return true;
  return false;
}

// One ordered pass over committed exact history. Verified exact-closure
// returns mark per-outlet usage at the assimilation chronology; dispatched
// outlet actions whose loop never closed are counted as unsettled, never as
// usage; verified returns on unregistered channels are counted and attribute
// to nothing.
__device__ inline void coping_observe_history(
    DirectCopingLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count) {
  if (ledger == nullptr || records == nullptr) return;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind == DirectExactHistoryKind::motor_output &&
        coping_outlet_registered(ledger, record.subject)) {
      bool settled = false;
      for (std::uint32_t j = 0u; j < count && !settled; ++j)
        settled = records[j].kind == DirectExactHistoryKind::world_return &&
                  records[j].identity == record.identity;
      if (!settled) ++ledger->unsettled_dispatches;
      continue;
    }
    if (record.kind != DirectExactHistoryKind::world_return ||
        (record.flags & kDirectHistoryVerifiedObservation) == 0u ||
        record.resource_delta != direct_adult_core::kQ16One)
      continue;
    DirectCopingOutlet* outlet =
        coping_find_outlet(ledger, record.subject);
    if (outlet == nullptr) {
      ++ledger->unregistered_returns;
      continue;
    }
    ++outlet->settled_actions;
    outlet->last_settled_tick = record.resident_tick;
  }
}

// Settled outlet actions across all registered outlets within the recency
// window ending at `tick`. Only exact verified closures on registered
// channels count; anything else composes to zero usage.
__device__ inline std::uint32_t coping_usage_in_window(
    const DirectCopingLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint32_t tick) {
  if (ledger == nullptr || ledger->count == 0u || records == nullptr)
    return 0u;
  const std::uint32_t earliest =
      tick > kCopingUsageWindowTicks ? tick - kCopingUsageWindowTicks : 0u;
  std::uint32_t usage = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::world_return ||
        (record.flags & kDirectHistoryVerifiedObservation) == 0u ||
        record.resource_delta != direct_adult_core::kQ16One ||
        record.resident_tick < earliest || record.resident_tick > tick)
      continue;
    usage += coping_outlet_registered(ledger, record.subject) ? 1u : 0u;
  }
  return usage;
}

// Bounded monotone attenuation for one stressor deposit: zero usage blunt
// nothing, each settled action adds a step, and the ceiling pins the law so
// coping can never invert into amplification.
__device__ inline std::int32_t coping_attenuation_q16(std::uint32_t usage) {
  const std::int32_t raw = static_cast<std::int32_t>(usage) *
                           kCopingPerActionBluntQ16;
  return raw > kCopingBluntCeilingQ16 ? kCopingBluntCeilingQ16 : raw;
}

// One participating stressor contact at device tick `tick`, composed from the
// parent's own primitives. The relent/recovery block, acute rise and contact
// bookkeeping run exactly as allostatic_contact runs them; the chronic path
// then deposits the ATTENUATED sample instead of the constant, with the
// sustained-sample accounting and drift gate matching the parent walk step
// for step. Zero attenuation reproduces the parent ledger byte-for-byte.
__device__ inline void coping_contact(DirectAllostaticLedger* ledger,
                                      DirectAllostaticEntry& entry,
                                      std::uint32_t tick, bool enable_acute,
                                      bool enable_chronic,
                                      std::int32_t blunt_q16) {
  const bool in_window = entry.contacts != 0u &&
                         tick - entry.last_contact_tick <=
                             kAllostaticWindowTicks;
  allostatic_contact(entry, tick, enable_acute, false);
  if (!enable_chronic) return;
  entry.sustained_samples = in_window ? entry.sustained_samples + 1u : 1u;
  const std::int32_t sample = kAllostaticLoadSampleQ16 -
                              direct_adult_core::mul_q16(
                                  kAllostaticLoadSampleQ16, blunt_q16);
  entry.load_accumulator_q16 =
      entry.load_accumulator_q16 + sample > kAllostaticLoadCeilingQ16
          ? kAllostaticLoadCeilingQ16
          : entry.load_accumulator_q16 + sample;
  if (entry.sustained_samples >= kAllostaticSustainedMinSamples &&
      entry.load_accumulator_q16 >= kAllostaticLoadSustainedQ16) {
    entry.learning_rate_q16 =
        entry.learning_rate_q16 - kAllostaticDriftStepQ16 <
                kAllostaticRateFloorQ16
            ? kAllostaticRateFloorQ16
            : entry.learning_rate_q16 - kAllostaticDriftStepQ16;
    entry.threshold_q16 =
        entry.threshold_q16 + kAllostaticDriftStepQ16 >
                kAllostaticThresholdCeilingQ16
            ? kAllostaticThresholdCeilingQ16
            : entry.threshold_q16 + kAllostaticDriftStepQ16;
  }
}

// One walk per stressor source over that source's exact history in record
// order, with each chronic deposit blunted by the settled outlet usage inside
// the recency window of that contact's resident tick.
__device__ inline void coping_learn_source(
    DirectAllostaticLedger* allostatic, const DirectCopingLedger* coping,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t subject, bool enable_acute, bool enable_chronic) {
  DirectAllostaticEntry* entry = allostatic_slot(allostatic, subject);
  if (entry == nullptr) return;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact ||
        record.subject != subject)
      continue;
    const std::uint32_t usage =
        coping_usage_in_window(coping, records, count, record.resident_tick);
    coping_contact(allostatic, *entry, record.resident_tick, enable_acute,
                   enable_chronic, coping_attenuation_q16(usage));
  }
  entry->disposition = allostatic_classify(*entry);
}

}  // namespace substrate::direct_network

#endif
