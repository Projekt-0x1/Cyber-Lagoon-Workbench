#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DOMAIN_PHENOTYPES_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DOMAIN_PHENOTYPES_CUH

// f.domain_phenotypes (#1560): specialized behavioral and computational
// phenotypes that emerge from distinct environmental, bodily and resource
// histories.
//
// Law anchors (Revision 12):
//   * emergence never means unauthored physics: a domain phenotype grows only
//     from explicit deterministic local laws applied to device-owned exact
//     history -- coherent environmental co-occurrence, lived bodily
//     consequence fate carried by settled world returns, and the resource
//     those consequences spent. No host label participates anywhere;
//   * formation demands lived consequence, not mere exposure: a candidate
//     consolidates only when its member pairs crossed both the binding and
//     the fate-purity gate. Histories that spend resources without a coherent
//     consequence fate refuse to consolidate, and the refusal is counted;
//   * there is no language predicate in this file. Any surface channel earns
//     membership through exactly the same environmental/bodily/resource laws;
//     a channel that never participates changes nothing, and removing one
//     later removes only the bindings that carry it while the surviving
//     domain keeps its learned core and its nonlinguistic behavioral record.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDomainPairCapacity = 256u;
inline constexpr std::uint32_t kDomainCapacity = 32u;
inline constexpr std::uint32_t kDomainLegCapacity = 6u;
inline constexpr std::uint32_t kDomainResponseCapacity = 8u;
inline constexpr std::uint32_t kDomainRecentCapacity = 64u;
inline constexpr std::uint32_t kDomainCoherenceWindowTicks = 16u;
inline constexpr std::uint32_t kDomainConsequenceWindowTicks = 24u;
inline constexpr std::uint32_t kDomainResponseWindowTicks = 12u;
inline constexpr std::int32_t kDomainRiseGainQ16 = direct_adult_core::kQ16One / 2;
inline constexpr std::int32_t kDomainFateGainQ16 = direct_adult_core::kQ16One / 3;
inline constexpr std::int32_t kDomainBindThresholdQ16 =
    (direct_adult_core::kQ16One * 3) / 4;
inline constexpr std::int32_t kDomainFateThresholdQ16 =
    direct_adult_core::kQ16One / 2;

struct DirectDomainContact {
  std::uint32_t channel;
  std::uint32_t value;
  std::uint32_t resident_tick;
};

struct DirectDomainPair {
  std::uint32_t channel_a;
  std::uint32_t value_a;
  std::uint32_t channel_b;
  std::uint32_t value_b;
  std::uint32_t cooccurrences;
  std::int32_t bind_mass_q16;
  std::int32_t approach_mass_q16;
  std::int32_t avoid_mass_q16;
  std::int32_t resource_spend;
  std::uint32_t first_tick;
  std::uint32_t last_tick;
};

struct DirectDomainResponse {
  std::uint32_t value;
  std::uint32_t count;
};

struct DirectDomain {
  DirectDomainContact legs[kDomainLegCapacity];
  std::uint32_t leg_count;
  std::int32_t closure_mass_q16;
  std::int32_t approach_mass_q16;
  std::int32_t avoid_mass_q16;
  std::int32_t resource_spend;
  std::uint32_t vital_actions;
  std::uint32_t damaged_actions;
  std::uint32_t activations;
  DirectDomainResponse responses[kDomainResponseCapacity];
  std::uint32_t response_count;
  std::uint32_t first_tick;
  std::uint32_t last_tick;
};
static_assert(std::is_trivial_v<DirectDomain> &&
              std::is_standard_layout_v<DirectDomain> &&
              std::has_unique_object_representations_v<DirectDomain>);

struct DirectDomainPhenotypeTable {
  DirectDomainPair pairs[kDomainPairCapacity];
  std::uint32_t pair_count;
  DirectDomain domains[kDomainCapacity];
  std::uint32_t domain_count;
  std::uint32_t fence_refusals;
  std::uint32_t fate_conflict_refusals;
  std::uint32_t consequences_processed;
};
static_assert(std::is_trivial_v<DirectDomainPhenotypeTable> &&
              std::is_standard_layout_v<DirectDomainPhenotypeTable> &&
              std::has_unique_object_representations_v<DirectDomainPhenotypeTable>);

__device__ inline std::int32_t domain_rise_q16(std::int32_t mass,
                                               std::int32_t gain_q16) {
  const std::int32_t headroom = direct_adult_core::kQ16One - mass;
  return mass + direct_adult_core::mul_q16(headroom > 0 ? headroom : 0,
                                           gain_q16);
}

__device__ inline bool domain_pair_bound(const DirectDomainPair& pair) {
  return pair.bind_mass_q16 >= kDomainBindThresholdQ16;
}

__device__ inline std::int32_t domain_minority_mass(const DirectDomainPair& pair) {
  return pair.approach_mass_q16 < pair.avoid_mass_q16
             ? direct_adult_core::max_q16(pair.approach_mass_q16, 0)
             : direct_adult_core::max_q16(pair.avoid_mass_q16, 0);
}

__device__ inline std::int32_t domain_dominant_mass(const DirectDomainPair& pair) {
  return direct_adult_core::max_q16(pair.approach_mass_q16,
                                    pair.avoid_mass_q16);
}

// Fate purity: the pair must have lived consequences and they must agree in
// valence. Mixed damage/vitality histories leave both masses past the gate
// and the pair refuses to carry a domain.
__device__ inline bool domain_pair_fate_pure(const DirectDomainPair& pair) {
  return domain_dominant_mass(pair) >= kDomainFateThresholdQ16 &&
         domain_minority_mass(pair) < kDomainFateThresholdQ16;
}

__device__ inline bool domain_leg_present(const DirectDomain& domain,
                                          std::uint32_t channel,
                                          std::uint32_t value) {
  for (std::uint32_t i = 0u; i < domain.leg_count; ++i)
    if (domain.legs[i].channel == channel && domain.legs[i].value == value)
      return true;
  return false;
}

__device__ inline std::int32_t domain_find_pair(
    const DirectDomainPhenotypeTable* table, std::uint32_t channel_a,
    std::uint32_t value_a, std::uint32_t channel_b, std::uint32_t value_b) {
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectDomainPair& pair = table->pairs[i];
    if (pair.channel_a == channel_a && pair.value_a == value_a &&
        pair.channel_b == channel_b && pair.value_b == value_b)
      return static_cast<std::int32_t>(i);
  }
  return -1;
}

// Environmental axis: contacts on distinct channels inside one coherence
// window deposit one evidence unit. Payload bytes stay opaque; only channel,
// value and resident tick from device exact history participate.
__device__ inline void domain_deposit_cooccurrence(
    DirectDomainPhenotypeTable* table, const DirectDomainContact& a,
    const DirectDomainContact& b) {
  if (a.channel == b.channel) return;
  const bool a_first = a.channel < b.channel ||
                       (a.channel == b.channel && a.value <= b.value);
  const DirectDomainContact& lo = a_first ? a : b;
  const DirectDomainContact& hi = a_first ? b : a;
  const std::int32_t slot =
      domain_find_pair(table, lo.channel, lo.value, hi.channel, hi.value);
  if (slot >= 0) {
    DirectDomainPair& pair = table->pairs[slot];
    ++pair.cooccurrences;
    pair.bind_mass_q16 = domain_rise_q16(pair.bind_mass_q16,
                                         kDomainRiseGainQ16);
    pair.last_tick = hi.resident_tick;
    return;
  }
  if (table->pair_count >= kDomainPairCapacity) {
    ++table->fence_refusals;
    return;
  }
  DirectDomainPair fresh{};
  fresh.channel_a = lo.channel;
  fresh.value_a = lo.value;
  fresh.channel_b = hi.channel;
  fresh.value_b = hi.value;
  fresh.cooccurrences = 1u;
  fresh.bind_mass_q16 = domain_rise_q16(0, kDomainRiseGainQ16);
  fresh.first_tick = lo.resident_tick;
  fresh.last_tick = hi.resident_tick;
  table->pairs[table->pair_count++] = fresh;
}

__device__ inline void domain_ingest_environment(
    DirectDomainPhenotypeTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  DirectDomainContact recent[kDomainRecentCapacity];
  std::uint32_t recent_count = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    const DirectDomainContact contact{record.subject, record.value,
                                      record.resident_tick};
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick -
                                          recent[j].resident_tick
                                    : 0u;
      if (age > kDomainCoherenceWindowTicks) continue;
      domain_deposit_cooccurrence(table, recent[j], contact);
    }
    std::uint32_t kept = 0u;
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick -
                                          recent[j].resident_tick
                                    : 0u;
      if (age <= kDomainCoherenceWindowTicks && kept < kDomainRecentCapacity)
        recent[kept++] = recent[j];
    }
    if (kept < kDomainRecentCapacity) recent[kept++] = contact;
    recent_count = kept;
  }
}

// Bodily axis: one settled verified world return raises one valence mass on
// every candidate pair whose exact legs both participated inside the
// consequence window before the originating action's emission, and books the
// return's absolute resource delta as those pairs' spent matter.
__device__ inline bool domain_credit_fate(
    DirectDomainPhenotypeTable* table,
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
  if (!found_emission) return false;
  bool seen_a[kDomainPairCapacity] = {};
  bool seen_b[kDomainPairCapacity] = {};
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    if (record.resident_tick > emission_tick) continue;
    if (emission_tick - record.resident_tick > kDomainConsequenceWindowTicks)
      continue;
    for (std::uint32_t p = 0u; p < table->pair_count; ++p) {
      const DirectDomainPair& pair = table->pairs[p];
      if (record.subject == pair.channel_a && record.value == pair.value_a)
        seen_a[p] = true;
      else if (record.subject == pair.channel_b &&
               record.value == pair.value_b)
        seen_b[p] = true;
    }
  }
  const bool vital =
      (world_return.flags & kDirectHistoryPayloadFlags) == 0u &&
      world_return.resource_delta >= 0;
  const std::int64_t absolute_delta =
      world_return.resource_delta < 0 ? -world_return.resource_delta
                                      : world_return.resource_delta;
  for (std::uint32_t p = 0u; p < table->pair_count; ++p) {
    if (!seen_a[p] || !seen_b[p]) continue;
    DirectDomainPair& pair = table->pairs[p];
    if (vital)
      pair.approach_mass_q16 =
          domain_rise_q16(pair.approach_mass_q16, kDomainFateGainQ16);
    else
      pair.avoid_mass_q16 =
          domain_rise_q16(pair.avoid_mass_q16, kDomainFateGainQ16);
    pair.resource_spend += static_cast<std::int32_t>(absolute_delta);
  }
  return true;
}

__device__ inline void domain_ingest_consequences(
    DirectDomainPhenotypeTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::world_return) continue;
    if ((records[i].flags & kDirectHistoryVerifiedObservation) == 0u) {
      ++table->fence_refusals;
      continue;
    }
    ++table->consequences_processed;
    domain_credit_fate(table, records, count, records[i]);
  }
}

__device__ inline void domain_absorb_pair(DirectDomain& domain,
                                          const DirectDomainPair& pair) {
  const DirectDomainContact legs[2] = {
      {pair.channel_a, pair.value_a, pair.first_tick},
      {pair.channel_b, pair.value_b, pair.last_tick}};
  for (std::uint32_t m = 0u; m < 2u; ++m) {
    if (domain_leg_present(domain, legs[m].channel, legs[m].value)) continue;
    if (domain.leg_count >= kDomainLegCapacity) break;
    std::uint32_t insert = domain.leg_count;
    while (insert > 0u &&
           (domain.legs[insert - 1u].channel > legs[m].channel ||
            (domain.legs[insert - 1u].channel == legs[m].channel &&
             domain.legs[insert - 1u].value > legs[m].value))) {
      domain.legs[insert] = domain.legs[insert - 1u];
      --insert;
    }
    domain.legs[insert] = legs[m];
    ++domain.leg_count;
  }
  if (pair.bind_mass_q16 < domain.closure_mass_q16)
    domain.closure_mass_q16 = pair.bind_mass_q16;
  domain.approach_mass_q16 += pair.approach_mass_q16;
  domain.avoid_mass_q16 += pair.avoid_mass_q16;
  domain.resource_spend += pair.resource_spend;
  if (pair.first_tick < domain.first_tick) domain.first_tick = pair.first_tick;
  if (pair.last_tick > domain.last_tick) domain.last_tick = pair.last_tick;
}

// Consolidation: bound and fate-pure pairs merge transitively over shared
// exact legs into one domain apiece. Deterministic order: domains follow
// their smallest member-pair index; legs sort by (channel, value).
__device__ inline void domain_extract(DirectDomainPhenotypeTable* table) {
  table->domain_count = 0u;
  bool merged[kDomainPairCapacity] = {};
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectDomainPair& seed = table->pairs[i];
    if (!domain_pair_bound(seed) || !domain_pair_fate_pure(seed) || merged[i])
      continue;
    merged[i] = true;
    DirectDomain domain{};
    domain.closure_mass_q16 = direct_adult_core::kQ16One;
    domain.first_tick = 0xffffffffu;
    domain_absorb_pair(domain, seed);
    bool grew = true;
    while (grew) {
      grew = false;
      for (std::uint32_t j = 0u; j < table->pair_count; ++j) {
        const DirectDomainPair& pair = table->pairs[j];
        if (!domain_pair_bound(pair) || !domain_pair_fate_pure(pair) ||
            merged[j])
          continue;
        bool joins = false;
        for (std::uint32_t l = 0u; l < domain.leg_count && !joins; ++l)
          joins = (pair.channel_a == domain.legs[l].channel &&
                   pair.value_a == domain.legs[l].value) ||
                  (pair.channel_b == domain.legs[l].channel &&
                   pair.value_b == domain.legs[l].value);
        if (!joins) continue;
        merged[j] = true;
        domain_absorb_pair(domain, pair);
        grew = true;
      }
    }
    if (table->domain_count >= kDomainCapacity) {
      ++table->fence_refusals;
      return;
    }
    table->domains[table->domain_count++] = domain;
  }
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectDomainPair& pair = table->pairs[i];
    if (merged[i]) continue;
    const std::int32_t minority = domain_minority_mass(pair);
    if (domain_dominant_mass(pair) >= kDomainFateThresholdQ16 &&
        minority >= kDomainFateThresholdQ16)
      ++table->fate_conflict_refusals;
  }
}

// Remove every binding that carries one channel and reform the domains from
// the survivors. Surviving pairs keep their learned bytes; the reformed
// domains keep theirs except for the removed channel's contributions.
__device__ inline std::uint32_t domain_dissolve_channel(
    DirectDomainPhenotypeTable* table, std::uint32_t channel) {
  std::uint32_t kept = 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < table->pair_count; ++i) {
    const DirectDomainPair& pair = table->pairs[i];
    if (pair.channel_a == channel || pair.channel_b == channel) {
      ++removed;
      continue;
    }
    table->pairs[kept++] = pair;
  }
  table->pair_count = kept;
  domain_extract(table);
  return removed;
}

__device__ inline std::int32_t domain_with_leg(
    const DirectDomainPhenotypeTable* table, std::uint32_t channel,
    std::uint32_t value) {
  for (std::uint32_t i = 0u; i < table->domain_count; ++i)
    if (domain_leg_present(table->domains[i], channel, value))
      return static_cast<std::int32_t>(i);
  return -1;
}

namespace domain_detail {

struct DomainActivationScan {
  const DirectExactHistoryRecord* records;
  std::uint32_t count;
};

// True when one exact leg of the given set was contacted within the response
// window before the emission tick.
__device__ inline bool domain_activated_before(
    const DomainActivationScan& scan,
    const DirectDomainContact* legs, std::uint32_t leg_count,
    std::uint32_t emission_tick) {
  for (std::uint32_t i = 0u; i < scan.count; ++i) {
    const DirectExactHistoryRecord& record = scan.records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    if (record.resident_tick > emission_tick) continue;
    if (emission_tick - record.resident_tick > kDomainResponseWindowTicks)
      continue;
    for (std::uint32_t l = 0u; l < leg_count; ++l)
      if (record.subject == legs[l].channel && record.value == legs[l].value)
        return true;
  }
  return false;
}

}  // namespace domain_detail

// Computational phenotype: the subject's own emitted values in the wake of
// domain activations, measured from device history. Recomputation over a
// dissolved domain's surviving legs yields exactly the contributions those
// legs caused, so removal of one surface cannot rewrite the rest.
__device__ inline void domain_measure_responses(
    DirectDomain& domain, const DirectExactHistoryRecord* records,
    std::uint32_t count) {
  domain.response_count = 0u;
  domain.activations = 0u;
  const domain_detail::DomainActivationScan scan{records, count};
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::motor_output) continue;
    const std::uint32_t emission_tick = records[i].resident_tick;
    if (!domain_detail::domain_activated_before(scan, domain.legs,
                                                domain.leg_count,
                                                emission_tick))
      continue;
    ++domain.activations;
    bool present = false;
    for (std::uint32_t r = 0u; r < domain.response_count && !present; ++r)
      present = domain.responses[r].value == records[i].value;
    if (present) continue;
    if (domain.response_count >= kDomainResponseCapacity) continue;
    domain.responses[domain.response_count++] =
        DirectDomainResponse{records[i].value, 0u};
  }
}

// Behavioral phenotype: a settled verified world return counts as a lived
// action of every domain whose exact legs activated the return's own emission
// within the response window -- the same chronology discipline fate crediting
// uses, anchored to consequences the world actually settled.
__device__ inline bool domain_return_activated(
    const DirectDomain& domain, const DirectExactHistoryRecord* records,
    std::uint32_t count, const DirectExactHistoryRecord& world_return) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::motor_output) continue;
    if (records[i].identity != world_return.identity) continue;
    const domain_detail::DomainActivationScan scan{records, count};
    return domain_detail::domain_activated_before(scan, domain.legs,
                                                  domain.leg_count,
                                                  records[i].resident_tick);
  }
  return false;
}

__device__ inline void domain_measure_actions(
    DirectDomainPhenotypeTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  for (std::uint32_t d = 0u; d < table->domain_count; ++d) {
    table->domains[d].vital_actions = 0u;
    table->domains[d].damaged_actions = 0u;
  }
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::world_return) continue;
    if ((records[i].flags & kDirectHistoryVerifiedObservation) == 0u) continue;
    const bool vital =
        (records[i].flags & kDirectHistoryPayloadFlags) == 0u &&
        records[i].resource_delta >= 0;
    for (std::uint32_t d = 0u; d < table->domain_count; ++d) {
      if (!domain_return_activated(table->domains[d], records, count,
                                   records[i]))
        continue;
      if (vital)
        ++table->domains[d].vital_actions;
      else
        ++table->domains[d].damaged_actions;
    }
  }
}

// One deterministic growth pass: environment, consequences, consolidation,
// behavior. Host composition is zero -- the caller hands over device exact
// history and reads back the finished ledger.
__device__ inline void domain_grow_ledger(DirectDomainPhenotypeTable* table,
                                          const DirectExactHistoryRecord*
                                              records,
                                          std::uint32_t count) {
  domain_ingest_environment(table, records, count);
  domain_ingest_consequences(table, records, count);
  domain_extract(table);
  for (std::uint32_t d = 0u; d < table->domain_count; ++d)
    domain_measure_responses(table->domains[d], records, count);
  domain_measure_actions(table, records, count);
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_DOMAIN_PHENOTYPES_CUH
