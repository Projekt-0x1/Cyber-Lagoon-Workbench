#ifndef HARDWARE_NATIVE_DIRECT_ADULT_VICARIOUS_RESONANCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_VICARIOUS_RESONANCE_CUH

// f.vicarious_affective_resonance (#1541): mirroring of observed partner
// somatic state without confounding self-agency with other-agency.
//
// Law anchors (Revision 12):
//   * agency separation: a somatic sample observed on a mirror channel is
//     OTHER-agency only when this subject's own exact history holds no
//     efference copy for its participation identity. A sample whose identity
//     matches one of our own motor emissions is reafference -- it belongs to
//     the self-ledger path (#1517) and is refused from the mirror with
//     counted refusals. Classification keys on device-owned identities,
//     never on payload resemblance;
//   * value equality is not identity: mirrored activations travel with their
//     agency tag and the observed partner participation identity through every
//     downstream read. A self-generated state carrying byte-identical values
//     still reads as a different object;
//   * no self-attribution: mirrored (other-agency) activation cannot enter the
//     self-homeostatic affect table or the allostatic load ledger. The only
//     lawful route is the guarded deposit gate, which refuses other-agency and
//     unknown provenance outright and counts every refusal;
//   * specificity fence: a mirror lane mints only behind device evidence of a
//     real sensory contact on that channel (the #1517 innervation fence).
//     Fabricated history claims and membrane-rejected channels resonate at
//     nothing;
//   * fields are not a second brain: resonance modulates nothing by itself.
//     It accumulates saturating evidence levels and relaxes homeostatically
//     toward neutral when a lane goes quiet.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_allostatic_metaplasticity.cuh"
#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kVicariousMirrorLaneCapacity = 16u;
inline constexpr std::int32_t kVicariousRiseGainQ16 =
    direct_adult_core::kQ16One / 4;
inline constexpr std::int32_t kVicariousRelaxGainQ16 =
    direct_adult_core::kQ16One / 8;

enum class VicariousAgency : std::uint32_t {
  none = 0u,
  self_endogenous = 1u,
  observed_other = 2u,
};

struct alignas(8) DirectVicariousMirrorLane {
  std::uint64_t partner_identity;
  std::uint32_t channel;
  std::uint32_t mirrored_samples;
  std::int32_t level_q16;
  VicariousAgency agency;
};
static_assert(std::is_trivial_v<DirectVicariousMirrorLane> &&
              std::is_standard_layout_v<DirectVicariousMirrorLane> &&
              std::has_unique_object_representations_v<
                  DirectVicariousMirrorLane>);

struct alignas(8) DirectVicariousMirrorState {
  DirectVicariousMirrorLane lanes[kVicariousMirrorLaneCapacity];
  std::uint64_t records_consumed;
  std::uint64_t integrations;
  std::uint32_t count;
  std::uint32_t agency_refusals;
  std::uint32_t coupling_refusals;
  std::uint32_t bleed_refusals;
  std::uint32_t relaxations;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectVicariousMirrorState> &&
              std::is_standard_layout_v<DirectVicariousMirrorState> &&
              std::has_unique_object_representations_v<
                  DirectVicariousMirrorState>);

struct alignas(8) DirectVicariousReadout {
  std::uint64_t partner_identity;
  std::uint32_t channel;
  std::uint32_t mirrored_samples;
  std::int32_t level_q16;
  VicariousAgency agency;
};
static_assert(std::is_trivial_v<DirectVicariousReadout> &&
              std::is_standard_layout_v<DirectVicariousReadout> &&
              std::has_unique_object_representations_v<
                  DirectVicariousReadout>);

// Efference-copy test over this subject's exact history: a sample is
// SELF-caused exactly when one of our own motor emissions carries its
// participation identity. Zero identity is nobody's emission.
__device__ inline bool vicarious_sample_is_self_agency(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t ticket_identity) {
  if (ticket_identity == 0u) return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::motor_output &&
        records[i].identity == ticket_identity)
      return true;
  return false;
}

// Observed payload intensity in Q16: the ingress word's low half is the
// somatic magnitude; the high half stays available to transport framing.
__device__ inline std::int32_t vicarious_sample_intensity_q16(
    std::uint32_t word) {
  return static_cast<std::int32_t>(word & 0xFFFFu);
}

__device__ inline DirectVicariousMirrorLane* vicarious_lane_slot(
    DirectVicariousMirrorState* state,
    const DirectExactHistoryRecord* records, std::uint32_t record_count,
    std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->lanes[i].channel == channel) return &state->lanes[i];
  if (!affect_channel_innervated(records, record_count, channel)) {
    ++state->coupling_refusals;
    return nullptr;
  }
  if (state->count >= kVicariousMirrorLaneCapacity) {
    ++state->coupling_refusals;
    return nullptr;
  }
  DirectVicariousMirrorLane fresh{};
  fresh.channel = channel;
  state->lanes[state->count] = fresh;
  return &state->lanes[state->count++];
}

// One integration pass over a committed exact-history span. Only sensory
// contacts inside the declared somatic window are mirror material; each
// sample classifies by efference evidence before it can move any level, so
// our own echoes never masquerade as a partner's state. Lanes without fresh
// samples relax toward neutral -- silence is recovery, not an authored
// repair. One thread walks records in slot order so accumulation is
// deterministic regardless of scheduling.
__device__ inline void vicarious_integrate_records(
    DirectVicariousMirrorState* state, const DirectExactHistoryRecord* records,
    std::uint32_t record_count, std::uint32_t sample_start,
    std::uint32_t somatic_lo, std::uint32_t somatic_hi) {
  if (state == nullptr || records == nullptr) return;
  bool touched[kVicariousMirrorLaneCapacity]{};
  for (std::uint32_t i = sample_start; i < record_count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    const std::uint32_t channel = record.subject;
    if (channel < somatic_lo || channel >= somatic_hi) continue;
    if (vicarious_sample_is_self_agency(records, record_count,
                                        record.identity)) {
      ++state->agency_refusals;
      continue;
    }
    DirectVicariousMirrorLane* lane =
        vicarious_lane_slot(state, records, record_count, channel);
    if (lane == nullptr) continue;
    const std::int32_t intensity =
        vicarious_sample_intensity_q16(record.value);
    const std::int32_t headroom = direct_adult_core::kQ16One - lane->level_q16;
    if (intensity > 0 && headroom > 0)
      lane->level_q16 += direct_adult_core::mul_q16(
          headroom,
          direct_adult_core::mul_q16(kVicariousRiseGainQ16, intensity));
    lane->partner_identity = record.identity;
    lane->agency = VicariousAgency::observed_other;
    ++lane->mirrored_samples;
    for (std::uint32_t s = 0u; s < state->count; ++s)
      touched[s] |= state->lanes[s].channel == channel;
  }
  for (std::uint32_t s = 0u; s < state->count; ++s) {
    if (touched[s]) continue;
    DirectVicariousMirrorLane& lane = state->lanes[s];
    const std::int32_t relaxed =
        lane.level_q16 -
        direct_adult_core::mul_q16(lane.level_q16, kVicariousRelaxGainQ16);
    if (relaxed != lane.level_q16) {
      ++state->relaxations;
      lane.level_q16 = relaxed;
    }
  }
}

// Cursor-tracked page entry: re-running over a grown hot page deposits each
// sample exactly once.
__device__ inline void vicarious_integrate_mirror(
    DirectVicariousMirrorState* state,
    const DirectExactHistoryHotPage* page, std::uint32_t somatic_lo,
    std::uint32_t somatic_hi) {
  if (state == nullptr || page == nullptr) return;
  const std::uint32_t committed = page->committed_slots;
  const std::uint32_t start = static_cast<std::uint32_t>(
      state->records_consumed < committed ? state->records_consumed
                                          : committed);
  vicarious_integrate_records(state, page->records, committed, start,
                              somatic_lo, somatic_hi);
  state->records_consumed = committed;
  ++state->integrations;
}

__device__ inline const DirectVicariousMirrorLane* vicarious_find_lane(
    const DirectVicariousMirrorState* state, std::uint32_t channel) {
  if (state == nullptr) return nullptr;
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->lanes[i].channel == channel) return &state->lanes[i];
  return nullptr;
}

// The only downstream read surface: value and provenance travel together, so
// no consumer can strip the agency tag without producing a different object.
__device__ inline DirectVicariousReadout vicarious_read_activation(
    const DirectVicariousMirrorState* state, std::uint32_t channel) {
  DirectVicariousReadout readout{};
  readout.channel = channel;
  const DirectVicariousMirrorLane* lane = vicarious_find_lane(state, channel);
  if (lane == nullptr || lane->level_q16 <= 0) return readout;
  readout.level_q16 = lane->level_q16;
  readout.agency = lane->agency;
  readout.partner_identity = lane->partner_identity;
  readout.mirrored_samples = lane->mirrored_samples;
  return readout;
}

// The encoding of an equally-valued SELF-generated state on the same channel:
// identical value bytes, different object. Self states have no partner.
__device__ inline DirectVicariousReadout vicarious_self_readout(
    std::uint32_t channel, std::int32_t level_q16) {
  DirectVicariousReadout readout{};
  readout.channel = channel;
  readout.level_q16 = level_q16;
  readout.agency = VicariousAgency::self_endogenous;
  return readout;
}

// The guarded bridge into the self-homeostatic table. Other-agency and
// unclassified activation are refused with counted refusals: what a partner
// feels is not what this body has settled. Self-endogenous activation rides
// the ordinary #1517 vitality rise.
__device__ inline bool vicarious_homeostatic_deposit(
    DirectAffectBodyState* affect, DirectVicariousMirrorState* mirror,
    std::uint32_t channel, std::int32_t level_q16, VicariousAgency agency) {
  if (affect == nullptr || mirror == nullptr) return false;
  if (agency != VicariousAgency::self_endogenous) {
    ++mirror->bleed_refusals;
    return false;
  }
  const std::int32_t magnitude = direct_adult_core::clamp_q16(
      level_q16, 0, direct_adult_core::kQ16One);
  for (std::uint32_t i = 0u; i < affect->count; ++i) {
    DirectAffectBodyEntry& entry = affect->entries[i];
    if (entry.channel != channel) continue;
    entry.vitality_q16 =
        affect_rise_q16(entry.vitality_q16,
                        direct_adult_core::mul_q16(kVicariousRiseGainQ16,
                                                   magnitude));
    ++entry.vitality_samples;
    return true;
  }
  return false;
}

// The same firewall over the chronic-load ledger: mirrored distress may not
// become this subject's allostatic burden.
__device__ inline bool vicarious_allostatic_deposit(
    DirectAllostaticLedger* ledger, DirectVicariousMirrorState* mirror,
    std::uint64_t source, std::int32_t load_q16, VicariousAgency agency) {
  if (ledger == nullptr || mirror == nullptr) return false;
  if (agency != VicariousAgency::self_endogenous ||
      load_q16 > kAllostaticLoadCeilingQ16) {
    ++mirror->bleed_refusals;
    return false;
  }
  DirectAllostaticEntry* entry = allostatic_slot(ledger, source);
  if (entry == nullptr) {
    ++mirror->bleed_refusals;
    return false;
  }
  entry->load_accumulator_q16 = direct_adult_core::clamp_q16(
      entry->load_accumulator_q16 + load_q16, 0, kAllostaticLoadCeilingQ16);
  ++entry->contacts;
  return true;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_VICARIOUS_RESONANCE_CUH
