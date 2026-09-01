#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ALLOSTATIC_METAPLASTICITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ALLOSTATIC_METAPLASTICITY_CUH

// d.allostatic_stress_metaplasticity (#1523): bifurcation of acute local
// plasticity from chronic multi-timescale learning-rate and threshold drift
// under sustained load.
//
// Law anchors (Revision 12 learning ecology §3-§5, development atlas NET17,
// canonical architecture §7 no-authority-laundering / §14 finite information):
//   * acute and chronic control paths are different state on different
//     timescales. The acute path answers each actually-participating contact
//     occurrence immediately; the chronic path integrates sustained load
//     density over a bounded window and opens only after a minimum duration of
//     in-window contacts;
//   * magnitude cannot substitute for duration: one contact contributes at
//     most one capped load sample however large its claimed size;
//   * chronic drift modulates parameters only -- learning rate down,
//     activation threshold up, both pinned by floor/ceiling. It never writes
//     acute plasticity state and never mints participation or eligibility;
//   * allostasis, not ratchet: when load relents past the window, the
//     accumulator decays and drifted parameters recover toward baseline in
//     exact recovery steps, so full relief restores baseline exactly;
//   * every deposit derives from device-owned exact-history records (resident
//     chronology and recorded ordering); host transport values never compose
//     allostatic state.

#include <cstdint>

#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

using direct_adult_core::kQ16One;
using direct_adult_core::mul_q16;

inline constexpr std::uint32_t kAllostaticLedgerCapacity = 256u;
// A contact sustains load only inside this many resident ticks of the previous
// one; a wider gap is load relenting.
inline constexpr std::uint32_t kAllostaticWindowTicks = 8u;
// Duration gate: drift needs this many consecutive in-window contacts, so load
// density alone can never open it.
inline constexpr std::uint32_t kAllostaticSustainedMinSamples = 6u;
inline constexpr std::int32_t kAllostaticAcuteGainQ16 = kQ16One / 4;
inline constexpr std::int32_t kAllostaticLoadSampleQ16 = kQ16One / 8;
inline constexpr std::int32_t kAllostaticLoadSustainedQ16 = kQ16One / 2;
inline constexpr std::int32_t kAllostaticLoadCeilingQ16 = kQ16One;
inline constexpr std::int32_t kAllostaticLoadRelaxQ16 = (kQ16One * 3) / 4;
// Chronic drift per gated contact, recovery per relenting gap. Recovery is an
// exact multiple of drift so full relief restores baseline without residue.
inline constexpr std::int32_t kAllostaticDriftStepQ16 = kQ16One / 64;
inline constexpr std::int32_t kAllostaticRecoveryStepQ16 = kQ16One / 32;
inline constexpr std::int32_t kAllostaticBaselineRateQ16 = kQ16One;
inline constexpr std::int32_t kAllostaticBaselineThresholdQ16 = kQ16One / 2;
inline constexpr std::int32_t kAllostaticRateFloorQ16 = kQ16One / 4;
inline constexpr std::int32_t kAllostaticThresholdCeilingQ16 = kQ16One;

enum class AllostaticDisposition : std::uint32_t {
  baseline = 0u,
  acutely_responding = 1u,
  chronically_loaded = 2u,
  recovering = 3u,
};

struct DirectAllostaticEntry {
  std::uint64_t source;
  // Acute path: occurrence-local plasticity response.
  std::int32_t acute_plasticity_q16;
  // Chronic path: sustained-load integrator and the parameters it drifts.
  std::int32_t load_accumulator_q16;
  std::int32_t learning_rate_q16;
  std::int32_t threshold_q16;
  std::uint32_t contacts;
  std::uint32_t sustained_samples;
  std::uint32_t last_contact_tick;
  std::uint32_t coupling_refusals;
  std::uint32_t recovery_events;
  AllostaticDisposition disposition;
};

struct DirectAllostaticLedger {
  DirectAllostaticEntry entries[kAllostaticLedgerCapacity];
  std::uint32_t count;
  std::uint32_t capacity_refusals;
  std::uint32_t capped_samples;
  std::uint32_t reserved;
};

__device__ inline std::int32_t allostatic_rise_q16(std::int32_t mass,
                                                   std::int32_t gain_q16) {
  const std::int32_t headroom = kQ16One - mass;
  return mass +
         direct_adult_core::mul_q16(headroom > 0 ? headroom : 0, gain_q16);
}

__device__ inline DirectAllostaticEntry* allostatic_find(
    DirectAllostaticLedger* ledger, std::uint64_t source) {
  if (ledger == nullptr) return nullptr;
  for (std::uint32_t i = 0u; i < ledger->count; ++i)
    if (ledger->entries[i].source == source) return &ledger->entries[i];
  return nullptr;
}

__device__ inline DirectAllostaticEntry* allostatic_slot(
    DirectAllostaticLedger* ledger, std::uint64_t source) {
  DirectAllostaticEntry* entry = allostatic_find(ledger, source);
  if (entry != nullptr) return entry;
  if (ledger->count >= kAllostaticLedgerCapacity) {
    ++ledger->capacity_refusals;
    return nullptr;
  }
  entry = &ledger->entries[ledger->count++];
  *entry = DirectAllostaticEntry{};
  entry->source = source;
  entry->learning_rate_q16 = kAllostaticBaselineRateQ16;
  entry->threshold_q16 = kAllostaticBaselineThresholdQ16;
  return entry;
}

// One participating contact at device tick `tick`. The two timescales update
// through independent code paths over independent state. A gap wider than the
// window relents load before anything else: the accumulator decays and
// parameters recover one step toward baseline.
__device__ inline void allostatic_contact(DirectAllostaticEntry& entry,
                                          std::uint32_t tick,
                                          bool enable_acute,
                                          bool enable_chronic) {
  if (entry.contacts != 0u &&
      tick - entry.last_contact_tick > kAllostaticWindowTicks) {
    entry.load_accumulator_q16 = direct_adult_core::mul_q16(
        entry.load_accumulator_q16, kAllostaticLoadRelaxQ16);
    std::int32_t rate = entry.learning_rate_q16;
    std::int32_t threshold = entry.threshold_q16;
    if (rate < kAllostaticBaselineRateQ16)
      rate = rate + kAllostaticRecoveryStepQ16 > kAllostaticBaselineRateQ16
                 ? kAllostaticBaselineRateQ16
                 : rate + kAllostaticRecoveryStepQ16;
    if (threshold > kAllostaticBaselineThresholdQ16)
      threshold =
          threshold - kAllostaticRecoveryStepQ16 < kAllostaticBaselineThresholdQ16
              ? kAllostaticBaselineThresholdQ16
              : threshold - kAllostaticRecoveryStepQ16;
    if (rate != entry.learning_rate_q16 || threshold != entry.threshold_q16) {
      entry.learning_rate_q16 = rate;
      entry.threshold_q16 = threshold;
      ++entry.recovery_events;
    }
    entry.sustained_samples = 0u;
  }
  if (enable_acute)
    entry.acute_plasticity_q16 = allostatic_rise_q16(entry.acute_plasticity_q16,
                                                     kAllostaticAcuteGainQ16);
  if (enable_chronic) {
    if (entry.contacts != 0u &&
        tick - entry.last_contact_tick <= kAllostaticWindowTicks)
      ++entry.sustained_samples;
    else
      entry.sustained_samples = 1u;
    entry.load_accumulator_q16 =
        entry.load_accumulator_q16 + kAllostaticLoadSampleQ16 >
                kAllostaticLoadCeilingQ16
            ? kAllostaticLoadCeilingQ16
            : entry.load_accumulator_q16 + kAllostaticLoadSampleQ16;
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
  ++entry.contacts;
  entry.last_contact_tick = tick;
}

__device__ inline AllostaticDisposition allostatic_classify(
    const DirectAllostaticEntry& entry) {
  if (entry.sustained_samples >= kAllostaticSustainedMinSamples &&
      entry.load_accumulator_q16 >= kAllostaticLoadSustainedQ16)
    return AllostaticDisposition::chronically_loaded;
  if (entry.learning_rate_q16 != kAllostaticBaselineRateQ16 ||
      entry.threshold_q16 != kAllostaticBaselineThresholdQ16)
    return AllostaticDisposition::recovering;
  if (entry.acute_plasticity_q16 > 0)
    return AllostaticDisposition::acutely_responding;
  return AllostaticDisposition::baseline;
}

// One walk per source over that source's exact history in record order.
// Deposits key on resident chronology and recorded ordering only.
__device__ inline void allostatic_learn_source(
    DirectAllostaticLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint64_t subject, bool enable_acute,
    bool enable_chronic) {
  DirectAllostaticEntry* entry = allostatic_slot(ledger, subject);
  if (entry == nullptr) return;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact ||
        record.subject != subject)
      continue;
    allostatic_contact(*entry, record.resident_tick, enable_acute,
                       enable_chronic);
  }
  entry->disposition = allostatic_classify(*entry);
}

// Magnitude cannot substitute for duration: a claimed load sample larger than
// one in-window contact lands as exactly one contact's worth, counted.
__device__ inline void allostatic_claim_load(DirectAllostaticLedger* ledger,
                                             DirectAllostaticEntry& entry,
                                             std::int32_t claimed_q16) {
  if (claimed_q16 > kAllostaticLoadSampleQ16) {
    ++ledger->capped_samples;
    claimed_q16 = kAllostaticLoadSampleQ16;
  }
  entry.load_accumulator_q16 =
      entry.load_accumulator_q16 + claimed_q16 > kAllostaticLoadCeilingQ16
          ? kAllostaticLoadCeilingQ16
          : entry.load_accumulator_q16 + claimed_q16;
}

// Cross-timescale authority gate. A plasticity consumer may take a chronic
// modulation factor only when the entry is ACTUALLY chronically loaded under
// its own sustained evidence, the request narrows rather than amplifies, and
// it claims no deeper depression than the earned drift. Everything else fails
// closed to the unmodulated baseline rate with the refusal counted on the
// requesting entry.
__device__ inline std::int32_t allostatic_metaplastic_gate(
    DirectAllostaticLedger* ledger, DirectAllostaticEntry& entry,
    std::int32_t requested_rate_q16) {
  const bool lawfully_chronic =
      entry.disposition == AllostaticDisposition::chronically_loaded;
  const bool narrowing = requested_rate_q16 <= kAllostaticBaselineRateQ16;
  const bool within_earned_drift =
      requested_rate_q16 >= entry.learning_rate_q16;
  if (!lawfully_chronic || !narrowing || !within_earned_drift) {
    ++entry.coupling_refusals;
    return kAllostaticBaselineRateQ16;
  }
  return requested_rate_q16;
}

}  // namespace substrate::direct_network

#endif
