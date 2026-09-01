#ifndef HARDWARE_NATIVE_DIRECT_ADULT_NEUROSCIENCE_COMPARANDA_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_NEUROSCIENCE_COMPARANDA_CUH

// h.neuroscience_comparandum (#1589).
// Observer-side benchmarking of measured network phenotypes against
// biological property bands.  Biology is a donor, not an ABI: bands are
// property-level observer handles with explicit provenance; verdicts never
// re-enter the resident subject.

#include <cmath>
#include <cstdint>

#include "hardware_native/direct_adult_posthoc_network_discovery.cuh"

namespace substrate::direct_adult_core {

struct DirectComparandumBand {
  const char* property;
  double low;
  double high;
  const char* provenance;
};

// Biological reference bands (property level, generous measurement bands).
//   CV of cortical spike-train irregularity ~0.5-1.2 (Softky & Koch 1993);
//   sparse activity fractions in cortex sit well below saturation;
//   resting functional connectivity is present but far from unity;
//   biological modules are subsets of the tissue, never the whole brain.
inline constexpr DirectComparandumBand kComparandumBands[4] = {
    {"firing_irregularity_cv", 0.3, 1.8,
     "cortical spike-train CV range (Softky & Koch 1993)"},
    {"activity_sparsity_fraction", 0.05, 0.95,
     "sparse non-saturating cortical activity fractions"},
    {"mean_functional_connectivity", 0.05, 0.85,
     "present-but-non-saturating resting functional connectivity"},
    {"largest_module_extent_fraction", 0.05, 0.6,
     "modular subsets, not whole-tissue collapse"},
};

inline constexpr std::uint32_t kComparandumBandCount = 4u;

struct DirectPhenotypeStatistics {
  double firing_irregularity_cv;
  double activity_sparsity_fraction;
  double mean_functional_connectivity;
  double largest_module_extent_fraction;
};

struct DirectComparandumVerdict {
  bool in_band[kComparandumBandCount];
  double observed[kComparandumBandCount];
};

// Statistics from measured traces plus the module report they produced.
inline DirectPhenotypeStatistics measure_network_phenotype_statistics(
    const DirectActivationTrace* traces, std::uint32_t trace_count,
    std::uint32_t node_count, const DirectPosthocDiscoveryReport& modules) {
  DirectPhenotypeStatistics stats{};
  if (traces == nullptr || trace_count < 2u || node_count < 2u)
    return stats;
  // Firing irregularity: mean per-node coefficient of variation of the
  // activation series across windows (active nodes only).
  double cv_sum = 0.0;
  std::uint32_t cv_nodes = 0u;
  double active_total = 0.0;
  for (std::uint32_t i = 0u; i < node_count && i < kDiscoveryMaxNodes; ++i) {
    double mean = 0.0;
    for (std::uint32_t t = 0u; t < trace_count; ++t)
      mean += static_cast<double>(traces[t].visits[i]);
    mean /= static_cast<double>(trace_count);
    if (mean <= 0.0) continue;
    ++cv_nodes;
    active_total += 1.0;
    double variance = 0.0;
    for (std::uint32_t t = 0u; t < trace_count; ++t) {
      const double d = static_cast<double>(traces[t].visits[i]) - mean;
      variance += d * d;
    }
    variance /= static_cast<double>(trace_count - 1u);
    cv_sum += std::sqrt(variance) / mean;
  }
  stats.firing_irregularity_cv =
      cv_nodes != 0u ? cv_sum / static_cast<double>(cv_nodes) : 0.0;
  // Sparsity: fraction of node-window samples that are silent.
  std::uint64_t silent = 0ull, samples = 0ull;
  for (std::uint32_t t = 0u; t < trace_count; ++t)
    for (std::uint32_t i = 0u; i < node_count && i < kDiscoveryMaxNodes; ++i) {
      ++samples;
      if (traces[t].visits[i] == 0u) ++silent;
    }
  stats.activity_sparsity_fraction =
      samples != 0ull ? static_cast<double>(silent) /
                            static_cast<double>(samples)
                      : 1.0;
  // Mean functional connectivity: mean pairwise |r| over varying nodes.
  {
    posthoc_detail::PairCohesion matrix{};
    std::uint64_t work = 0ull;
    posthoc_detail::measure_all_pairs(traces, trace_count,
                                      node_count < kDiscoveryMaxNodes
                                          ? node_count
                                          : kDiscoveryMaxNodes,
                                      &matrix, &work);
    double r_sum = 0.0;
    std::uint64_t pairs = 0ull;
    for (std::uint32_t i = 0u; i + 1u < node_count; ++i)
      for (std::uint32_t j = i + 1u; j < node_count; ++j) {
        r_sum += std::sqrt(
            static_cast<double>(matrix.cohesion[i][j]) / 65536.0);
        ++pairs;
      }
    stats.mean_functional_connectivity =
        pairs != 0ull ? r_sum / static_cast<double>(pairs) : 0.0;
  }
  // Modular extent: largest discovered module over the whole tissue.
  std::uint32_t largest = 0u;
  for (std::uint32_t m = 0u; m < modules.module_count; ++m)
    if (modules.modules[m].member_count > largest)
      largest = modules.modules[m].member_count;
  stats.largest_module_extent_fraction =
      static_cast<double>(largest) / static_cast<double>(node_count);
  return stats;
}

inline void verdicts_for_statistics(const DirectPhenotypeStatistics& stats,
                                    DirectComparandumVerdict* out) {
  if (out == nullptr) return;
  *out = DirectComparandumVerdict{};
  const double observed[kComparandumBandCount] = {
      stats.firing_irregularity_cv, stats.activity_sparsity_fraction,
      stats.mean_functional_connectivity,
      stats.largest_module_extent_fraction};
  for (std::uint32_t b = 0u; b < kComparandumBandCount; ++b) {
    out->observed[b] = observed[b];
    out->in_band[b] =
        observed[b] >= kComparandumBands[b].low &&
        observed[b] <= kComparandumBands[b].high;
  }
}

}  // namespace substrate::direct_adult_core

#endif
