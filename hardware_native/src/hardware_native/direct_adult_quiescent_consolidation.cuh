#ifndef HARDWARE_NATIVE_DIRECT_ADULT_QUIESCENT_CONSOLIDATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_QUIESCENT_CONSOLIDATION_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_network_recipe.hpp"

#if defined(__CUDACC__)
#define DIRECT_QUIESCENT_HD __host__ __device__
#else
#define DIRECT_QUIESCENT_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectQuiescentPageLedgerCapacity = 8u;

// One continuing subject line may consolidate only while its membrane is
// quiet: no pending ingress and no pending consequence return, both read as
// real device ring cursors published by transport and consumed by the ingest
// kernels.  Endogenous delayed sparse propagation is resident tissue
// dynamics, not contact arrival, and never gates consolidation.  No host
// assertion and no wall clock may open this gate.
struct DirectQuiescentGateObservation {
  std::uint32_t ingress_pending;
  std::uint32_t consequence_pending;
};

// Two declared consolidation windows with distinct effects.  The long span
// must strictly contain the short span's horizon: aging past the short window
// alone may never reach cold candidacy while the long window still holds a
// replay hit, and no configuration may declare the spans equal or inverted.
struct DirectQuiescentConsolidationConfig {
  std::uint32_t short_window_epochs;
  std::uint32_t long_window_epochs;
  std::uint32_t short_replay_budget;
  std::uint32_t long_replay_budget;
  std::uint32_t long_replay_period;
};

struct DirectQuiescentPageLedgerEntry {
  recipe::Root256 address;
  std::uint32_t epochs_since_replay_short;
  std::uint32_t epochs_since_replay_long;
  std::uint64_t demand_reads_since_epoch;
  std::uint64_t replay_hits_total;
};
static_assert(std::is_standard_layout_v<DirectQuiescentPageLedgerEntry> &&
              std::is_trivial_v<DirectQuiescentPageLedgerEntry> &&
              std::has_unique_object_representations_v<DirectQuiescentPageLedgerEntry>);

struct DirectQuiescentPageLedger {
  DirectQuiescentPageLedgerEntry entries[kDirectQuiescentPageLedgerCapacity];
  std::uint32_t count;
  std::uint32_t overflow;
};
static_assert(std::is_standard_layout_v<DirectQuiescentPageLedger> &&
              std::is_trivial_v<DirectQuiescentPageLedger> &&
              std::has_unique_object_representations_v<DirectQuiescentPageLedger>);

enum class DirectQuiescentConsolidationStatus : std::uint32_t {
  idle = 0u,
  gate_refused = 1u,
  replay_diverged = 2u,
  completed = 3u,
  invalid_state = 4u,
};

struct DirectQuiescentConsolidationReceipt {
  DirectQuiescentConsolidationStatus status;
  std::uint32_t reserved;
  std::uint64_t generation;
  std::uint64_t gate_refusals;
  std::uint64_t replay_divergences;
  std::uint64_t pages_replayed_short;
  std::uint64_t pages_replayed_long;
  std::uint64_t pages_kept_hot;
  std::uint64_t cold_candidates;
  std::uint64_t forced_eviction_refusals;
  std::uint64_t eligibility_expired_reclaimed;
  std::uint64_t eligibility_live_preserved;
};
static_assert(std::is_standard_layout_v<DirectQuiescentConsolidationReceipt> &&
              std::is_trivial_v<DirectQuiescentConsolidationReceipt> &&
              std::has_unique_object_representations_v<DirectQuiescentConsolidationReceipt>);

DIRECT_QUIESCENT_HD inline bool observe_quiescence(
    const std::uint32_t* ingress_consumed_head, const std::uint32_t* ingress_published_tail,
    const std::uint32_t* consequence_consumed_head, const std::uint32_t* consequence_published_tail,
    DirectQuiescentGateObservation* out) {
  if (out == nullptr || ingress_consumed_head == nullptr ||
      ingress_published_tail == nullptr || consequence_consumed_head == nullptr ||
      consequence_published_tail == nullptr)
    return false;
  *out = DirectQuiescentGateObservation{
      *ingress_published_tail - *ingress_consumed_head,
      *consequence_published_tail - *consequence_consumed_head};
  return out->ingress_pending == 0u && out->consequence_pending == 0u;
}

DIRECT_QUIESCENT_HD inline bool valid_consolidation_config(
    const DirectQuiescentConsolidationConfig& config) {
  return config.long_window_epochs > config.short_window_epochs &&
         config.short_replay_budget != 0u &&
         config.long_replay_budget != 0u &&
         config.short_replay_budget <= kDirectQuiescentPageLedgerCapacity &&
         config.long_replay_budget <= kDirectQuiescentPageLedgerCapacity &&
         config.long_replay_period != 0u;
}

// Gate arm.  Under any pending contact the whole transaction refuses
// fail-closed: every bank keeps its bytes except the refusal counter, so a
// refused consolidation is observationally nothing but a counted refusal.
// On an open gate the ledger ages by exactly one epoch -- saturation keeps
// the bounded state finite without a clock.
DIRECT_QUIESCENT_HD inline bool begin_quiescent_consolidation(
    const DirectQuiescentGateObservation& observation,
    const DirectQuiescentConsolidationConfig& config,
    DirectQuiescentPageLedger* ledger, DirectQuiescentConsolidationReceipt* receipt) {
  if (ledger == nullptr || receipt == nullptr ||
      !valid_consolidation_config(config)) {
    if (receipt != nullptr)
      receipt->status = DirectQuiescentConsolidationStatus::invalid_state;
    return false;
  }
  if (observation.ingress_pending != 0u || observation.consequence_pending != 0u) {
    receipt->status = DirectQuiescentConsolidationStatus::gate_refused;
    ++receipt->gate_refusals;
    return false;
  }
  constexpr std::uint32_t kSaturating = 0xffffffffu;
  for (std::uint32_t i = 0u; i < ledger->count; ++i) {
    DirectQuiescentPageLedgerEntry& entry = ledger->entries[i];
    entry.epochs_since_replay_short =
        entry.epochs_since_replay_short == kSaturating
            ? kSaturating
            : entry.epochs_since_replay_short + 1u;
    entry.epochs_since_replay_long = entry.epochs_since_replay_long == kSaturating
                                         ? kSaturating
                                         : entry.epochs_since_replay_long + 1u;
    entry.demand_reads_since_epoch = 0u;
  }
  receipt->status = DirectQuiescentConsolidationStatus::idle;
  return true;
}

// Replay arm.  A replayed window outcome that misses its recorded root aborts
// the entire transaction fail-closed; a verified short-scale hit refreshes the
// short age, a verified long-scale hit only the long age.  Replaying is
// accessibility work: it touches this ledger line and nothing else, so a
// hundred repeats mint no evidence and strengthen no support.
DIRECT_QUIESCENT_HD inline bool record_page_replay(
    DirectQuiescentPageLedgerEntry* entry, bool verified_against_recorded_root,
    bool long_scale, DirectQuiescentConsolidationReceipt* receipt) {
  if (entry == nullptr || receipt == nullptr ||
      receipt->status == DirectQuiescentConsolidationStatus::gate_refused)
    return false;
  if (!verified_against_recorded_root) {
    receipt->status = DirectQuiescentConsolidationStatus::replay_diverged;
    ++receipt->replay_divergences;
    return false;
  }
  if (long_scale) {
    entry->epochs_since_replay_long = 0u;
    ++receipt->pages_replayed_long;
  } else {
    entry->epochs_since_replay_short = 0u;
    ++receipt->pages_replayed_short;
  }
  ++entry->replay_hits_total;
  return true;
}

// Bounded replay economics.  The transaction owns a finite replay budget and
// spends it on pages the resident actually demanded since the last epoch,
// ties broken by address so identical roots always select identically.
// Demand orders the work; it never decides placement.
DIRECT_QUIESCENT_HD inline std::uint32_t select_replay_set(
    const DirectQuiescentPageLedger& ledger, std::uint32_t budget,
    std::uint32_t* selected_indices) {
  if (selected_indices == nullptr || budget == 0u) return 0u;
  std::uint32_t selected = 0u;
  const std::uint32_t pool = ledger.count < kDirectQuiescentPageLedgerCapacity
                                 ? ledger.count
                                 : kDirectQuiescentPageLedgerCapacity;
  bool taken[kDirectQuiescentPageLedgerCapacity] = {};
  for (std::uint32_t pick = 0u; pick < budget && pick < pool; ++pick) {
    std::int32_t best = -1;
    for (std::uint32_t i = 0u; i < pool; ++i) {
      if (taken[i]) continue;
      if (best < 0) {
        best = static_cast<std::int32_t>(i);
        continue;
      }
      const DirectQuiescentPageLedgerEntry& candidate = ledger.entries[i];
      const DirectQuiescentPageLedgerEntry& incumbent = ledger.entries[best];
      bool wins = candidate.demand_reads_since_epoch > incumbent.demand_reads_since_epoch;
      if (!wins && candidate.demand_reads_since_epoch == incumbent.demand_reads_since_epoch)
        for (std::uint32_t w = 0u; w < 8u; ++w) {
          if (candidate.address.word[w] != incumbent.address.word[w]) {
            wins = candidate.address.word[w] < incumbent.address.word[w];
            break;
          }
        }
      if (wins) best = static_cast<std::int32_t>(i);
    }
    if (best < 0) break;
    taken[best] = true;
    selected_indices[selected++] = static_cast<std::uint32_t>(best);
  }
  return selected;
}

// Placement arm.  Verdicts come only from the two replay ages: a page with no
// replayed access across BOTH declared windows becomes a cold-archive
// candidate; a page replayed in either window stays hot.  Raw demand and any
// recency counter are invisible here, so a host-style LRU reading cannot
// reproduce these verdicts.
DIRECT_QUIESCENT_HD inline void place_ledger_pages(
    const DirectQuiescentPageLedger& ledger,
    const DirectQuiescentConsolidationConfig& config,
    std::uint32_t* cold_verdicts, DirectQuiescentConsolidationReceipt* receipt) {
  const std::uint32_t pool = ledger.count < kDirectQuiescentPageLedgerCapacity
                                 ? ledger.count
                                 : kDirectQuiescentPageLedgerCapacity;
  for (std::uint32_t i = 0u; i < pool; ++i) {
    const DirectQuiescentPageLedgerEntry& entry = ledger.entries[i];
    const bool cold = entry.epochs_since_replay_short > config.short_window_epochs &&
                      entry.epochs_since_replay_long > config.long_window_epochs;
    if (cold_verdicts != nullptr) cold_verdicts[i] = cold ? 1u : 0u;
    if (cold)
      ++receipt->cold_candidates;
    else
      ++receipt->pages_kept_hot;
  }
}

// A forced host-style eviction of a page replayed within either declared
// window is refused outright; only economically dead pages may be moved.
DIRECT_QUIESCENT_HD inline bool force_demote_page(
    DirectQuiescentPageLedgerEntry* entry,
    const DirectQuiescentConsolidationConfig& config,
    DirectQuiescentConsolidationReceipt* receipt) {
  if (entry == nullptr || receipt == nullptr ||
      !valid_consolidation_config(config))
    return false;
  if (entry->epochs_since_replay_short <= config.short_window_epochs ||
      entry->epochs_since_replay_long <= config.long_window_epochs) {
    ++receipt->forced_eviction_refusals;
    return false;
  }
  return true;
}

// Economic cleanup arm over the eligibility bank: an expired endogenous
// shadow is reclaimed in place, a live one keeps every byte, and settled-but-
// retained evidence outside this bank is never scanned at all.
DIRECT_QUIESCENT_HD inline void cleanup_expired_eligibility(
    direct_adult_core::EligibilityRecord* records, std::uint32_t count,
    std::uint32_t resident_tick, DirectQuiescentConsolidationReceipt* receipt) {
  if (records == nullptr || receipt == nullptr) return;
  for (std::uint32_t i = 0u; i < count; ++i) {
    direct_adult_core::EligibilityRecord& record = records[i];
    if (record.live == 0u) continue;
    if (record.expiry_tick < resident_tick) {
      record.live = 0u;
      ++receipt->eligibility_expired_reclaimed;
    } else {
      ++receipt->eligibility_live_preserved;
    }
  }
}

DIRECT_QUIESCENT_HD inline void finish_quiescent_consolidation(
    DirectQuiescentConsolidationReceipt* receipt) {
  if (receipt == nullptr) return;
  if (receipt->status == DirectQuiescentConsolidationStatus::idle)
    receipt->status = DirectQuiescentConsolidationStatus::completed;
  ++receipt->generation;
}

}  // namespace substrate::direct_network

#undef DIRECT_QUIESCENT_HD

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_QUIESCENT_CONSOLIDATION_CUH
