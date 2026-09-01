#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PROSODY_CLOSURE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PROSODY_CLOSURE_CUH

// j.steinhauer_prosody_syntax_closure (#1552): electrophysiological lens
// demonstrating temporal prosodic phrasing as syntactic boundary delimiter.
//
// Law anchors (Revision 12):
//   * observer handle, device mechanism: "prosody" and "syntax" name an
//     interpretation of temporal signal structure, never a runtime branch.
//     What lives here is a measurement over device-owned exact history:
//     do resident relational closure events (settled world returns and
//     committed sparse credit) phase-lock to the boundary clock laid down
//     by temporally bounded contact phrases, or scatter time-invariantly;
//   * the phrase schedule is transport metadata, exactly like an arrival
//     instant: the host drove it, the resident never sees it, and nothing
//     below branches on payload content or meaning;
//   * no authority laundering: the lens reads committed exact-history
//     records and writes only its own receipt; causal ledgers stay
//     read-only, and the chance baseline is deterministic arithmetic over
//     the same observed support -- never invented evidence;
//   * delimitation is claimed only when the share of closures inside the
//     densest band of the phrase cycle beats every time-shuffled placement
//     of the same events over the same support, searched just as hard
//     (empirical p <= 1/null_samples), never by narration.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

constexpr std::uint32_t kProsodyMaxPeriodTicks = 1024u;

struct alignas(8) DirectProsodyPhraseSpan {
  std::uint32_t onset_tick;   // admission tick of the phrase's first contact
  std::uint32_t closure_tick; // first resident tick after its last contact
};
static_assert(std::is_trivial_v<DirectProsodyPhraseSpan> &&
              std::is_standard_layout_v<DirectProsodyPhraseSpan> &&
              std::has_unique_object_representations_v<
                  DirectProsodyPhraseSpan>);

struct alignas(8) DirectProsodyClosureReceipt {
  std::uint64_t closure_events;
  std::uint32_t period_ticks;
  std::uint32_t window_ticks;
  std::uint32_t aligned_events;
  std::uint32_t aligned_fraction_q16;
  std::uint32_t best_phase_tick;
  std::uint32_t null_max_aligned_fraction_q16;
  std::uint32_t null_samples;
  std::uint32_t support_first_tick;
  std::uint32_t support_last_tick;
  std::uint32_t first_closure_tick;
  std::uint32_t last_closure_tick;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectProsodyClosureReceipt> &&
              std::is_standard_layout_v<DirectProsodyClosureReceipt> &&
              std::has_unique_object_representations_v<
                  DirectProsodyClosureReceipt>);

__device__ inline std::uint64_t prosody_next_random(std::uint64_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

struct DirectProsodyBandShare {
  std::uint32_t events_in_band;
  std::uint32_t fraction_q16;
  std::uint32_t band_start;
};

// Densest circular band of `window` ticks around the phrase cycle. The band
// slides across the whole cycle for the observed stream and for every
// shuffled placement alike, so both sides pay for the search over positions.
__device__ inline DirectProsodyBandShare prosody_best_band(
    const std::uint32_t* phase_histogram, std::uint32_t period,
    std::uint32_t window, std::uint64_t events) {
  const std::uint32_t span = window < period ? window : period;
  std::uint32_t running = 0u;
  for (std::uint32_t i = 0u; i < span; ++i) running += phase_histogram[i];
  std::uint32_t best = running;
  std::uint32_t best_start = 0u;
  for (std::uint32_t start = 1u; start < period; ++start) {
    running += phase_histogram[(start + span - 1u) % period] -
               phase_histogram[start - 1u];
    if (running > best) {
      best = running;
      best_start = start;
    }
  }
  DirectProsodyBandShare share{};
  share.events_in_band = best;
  share.band_start = best_start;
  share.fraction_q16 = static_cast<std::uint32_t>(
      (best * 65536ull + events / 2u) / events);
  return share;
}

// Measure where relational closure actually happened. Closure events are the
// settled-return and committed-credit entries of the exact history, streamed
// straight from the ledger into a phase histogram over the boundary cycle.
// The chance baseline reshuffles the same event count uniformly over the
// same driven support and keeps the strictest sample, so "above chance"
// means above every chance placement.
__device__ inline bool prosody_measure_closures(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectProsodyPhraseSpan* spans, std::uint32_t span_count,
    std::uint32_t period_ticks, std::uint32_t band_window,
    std::uint32_t null_samples, std::uint64_t null_seed,
    DirectProsodyClosureReceipt* receipt) {
  if (receipt == nullptr || records == nullptr || spans == nullptr ||
      span_count == 0u || period_ticks == 0u ||
      period_ticks > kProsodyMaxPeriodTicks || band_window == 0u ||
      null_samples == 0u)
    return false;
  for (std::uint32_t i = 0u; i < span_count; ++i)
    if (spans[i].closure_tick <= spans[i].onset_tick) return false;

  const std::uint32_t origin = spans[0].closure_tick % period_ticks;
  std::uint32_t phase_histogram[kProsodyMaxPeriodTicks] = {};
  DirectProsodyClosureReceipt out{};
  std::uint64_t events = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::world_return &&
        record.kind != DirectExactHistoryKind::sparse_credit)
      continue;
    const std::uint32_t tick = record.resident_tick;
    ++phase_histogram[(tick % period_ticks + period_ticks - origin) %
                      period_ticks];
    if (events == 0u || tick < out.first_closure_tick)
      out.first_closure_tick = tick;
    if (events == 0u || tick > out.last_closure_tick)
      out.last_closure_tick = tick;
    ++events;
  }
  if (events == 0u) return false;
  out.closure_events = events;

  out.period_ticks = period_ticks;
  out.window_ticks = band_window;

  std::uint32_t support_first = spans[0].onset_tick;
  std::uint32_t support_last = spans[0].closure_tick;
  for (std::uint32_t i = 1u; i < span_count; ++i) {
    if (spans[i].onset_tick < support_first) support_first = spans[i].onset_tick;
    if (spans[i].closure_tick > support_last) support_last = spans[i].closure_tick;
  }
  out.support_first_tick = support_first;
  out.support_last_tick = support_last;

  const DirectProsodyBandShare observed =
      prosody_best_band(phase_histogram, period_ticks, band_window, events);
  out.aligned_events = observed.events_in_band;
  out.aligned_fraction_q16 = observed.fraction_q16;
  out.best_phase_tick = observed.band_start;

  std::uint64_t state = null_seed | 1ull;
  const std::uint32_t support_width = support_last - support_first + 1u;
  std::uint32_t null_max_share = 0u;
  std::uint32_t scratch[kProsodyMaxPeriodTicks];
  for (std::uint32_t sample = 0u; sample < null_samples; ++sample) {
    for (std::uint32_t p = 0u; p < period_ticks; ++p) scratch[p] = 0u;
    for (std::uint64_t i = 0u; i < events; ++i) {
      const std::uint32_t tick =
          support_first +
          static_cast<std::uint32_t>(prosody_next_random(&state) %
                                     support_width);
      ++scratch[(tick % period_ticks + period_ticks - origin) % period_ticks];
    }
    const DirectProsodyBandShare share = prosody_best_band(
        scratch, period_ticks, band_window, events);
    if (share.fraction_q16 > null_max_share)
      null_max_share = share.fraction_q16;
  }
  out.null_max_aligned_fraction_q16 = null_max_share;
  out.null_samples = null_samples;

  *receipt = out;
  return true;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_PROSODY_CLOSURE_CUH
