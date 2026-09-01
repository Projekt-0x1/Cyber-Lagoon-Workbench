#ifndef HARDWARE_NATIVE_DIRECT_ADULT_POSTHOC_NETWORK_DISCOVERY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_POSTHOC_NETWORK_DISCOVERY_CUH

// h.posthoc_network_discovery (#1582).
// Observer-only empirical discovery surface: opaque node activation traces
// are recorded from lived contact, functional modules and inter-module
// connectome edges are derived post-hoc from measured coactivation alone --
// no seed atlas, semantic label, or host answer row -- and every derivation
// reads only host copies, so resident nodes, routes, matter, exact history,
// and executor state cannot be mutated by discovery.

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kDiscoveryMaxNodes = 256u;
inline constexpr std::uint32_t kDiscoveryMaxTraces = 64u;
inline constexpr std::uint32_t kDiscoveryMaxModules = 16u;
// Minimum co-firing windows before a pair's correlation counts as evidence:
// sparse mostly-zero columns can correlate perfectly by chance.
inline constexpr std::uint32_t kDiscoveryMinPairSupport = 6u;

struct DirectActivationTrace {
  std::uint32_t visits[kDiscoveryMaxNodes];  // firings since prior window
  std::uint32_t resident_tick;
};
static_assert(std::is_trivial_v<DirectActivationTrace> &&
              std::is_standard_layout_v<DirectActivationTrace>);

struct DirectDiscoveredModule {
  std::uint32_t members[kDiscoveryMaxNodes];
  std::uint32_t member_count;
  std::int32_t internal_cohesion_q16;  // mean within-module pair cohesion
  std::uint64_t module_identity;       // content-addressed member fold
};

struct DirectModuleEdge {
  std::uint16_t module_a;
  std::uint16_t module_b;
  std::uint32_t coupled_pairs;
  std::int32_t mean_coactivation_q16;  // mean cross-boundary pair cohesion
};

struct DirectPosthocDiscoveryReport {
  DirectDiscoveredModule modules[kDiscoveryMaxModules];
  DirectModuleEdge edges[kDiscoveryMaxModules];
  std::uint32_t module_count;
  std::uint32_t edge_count;
  std::uint64_t work_units;    // bounded-work census for the whole derivation
  std::uint64_t samples_folded;
};

// Observer-side recording: sample one node snapshot into a per-node
// positive-activation trace value (Q8-scaled).  The slice is a host copy;
// the resident arena is never touched.
inline void record_posthoc_activation_trace(
    const direct_network::DirectNode* current,
    std::uint32_t node_count, std::uint32_t resident_tick,
    DirectActivationTrace* out) {
  *out = DirectActivationTrace{};
  out->resident_tick = resident_tick;
  const std::uint32_t bounded =
      node_count < kDiscoveryMaxNodes ? node_count : kDiscoveryMaxNodes;
  for (std::uint32_t i = 0u; i < bounded; ++i) {
    const std::int32_t activation = current[i].activation_q16;
    out->visits[i] =
        activation > 0 ? static_cast<std::uint32_t>(activation >> 8) : 0u;
  }
}

// Measured pairwise cohesion in Q16 as the squared correlation of
// window-to-window activation CHANGES: shared slow envelopes carry no
// evidence, only genuine co-variation does.  Returns floor(65536 * r^2).
inline std::int32_t posthoc_pair_cohesion_q16(
    const DirectActivationTrace* traces, std::uint32_t trace_count,
    std::uint32_t node_i, std::uint32_t node_j, std::uint64_t* work) {
  if (trace_count < 2u) return 0;
  std::int64_t dot = 0, energy_i = 0, energy_j = 0;
  for (std::uint32_t t = 1u; t < trace_count; ++t) {
    const std::int64_t di =
        static_cast<std::int64_t>(traces[t].visits[node_i]) -
        static_cast<std::int64_t>(traces[t - 1u].visits[node_i]);
    const std::int64_t dj =
        static_cast<std::int64_t>(traces[t].visits[node_j]) -
        static_cast<std::int64_t>(traces[t - 1u].visits[node_j]);
    dot += di * dj;
    energy_i += di * di;
    energy_j += dj * dj;
    ++*work;
  }
  if (energy_i == 0 || energy_j == 0) return 0;
  const long double r_squared =
      static_cast<long double>(dot) * static_cast<long double>(dot) /
      (static_cast<long double>(energy_i) * static_cast<long double>(energy_j));
  std::int32_t q16 = static_cast<std::int32_t>(r_squared * 65536.0L);
  if (q16 > 65536) q16 = 65536;
  return q16;
}

namespace posthoc_detail {

struct PairCohesion {
  std::int32_t cohesion[kDiscoveryMaxNodes][kDiscoveryMaxNodes];
  std::uint32_t support[kDiscoveryMaxNodes][kDiscoveryMaxNodes];
};

inline void measure_all_pairs(const DirectActivationTrace* traces,
                              std::uint32_t trace_count,
                              std::uint32_t node_count,
                              PairCohesion* matrix, std::uint64_t* work) {
  for (std::uint32_t i = 0u; i < node_count; ++i)
    matrix->cohesion[i][i] = 1 << 16;
  for (std::uint32_t i = 0u; i + 1u < node_count; ++i)
    for (std::uint32_t j = i + 1u; j < node_count; ++j) {
      std::uint32_t support = 0u;
      for (std::uint32_t t = 0u; t < trace_count; ++t) {
        if (traces[t].visits[i] != 0u && traces[t].visits[j] != 0u) ++support;
        ++*work;
      }
      matrix->support[i][j] = support;
      matrix->support[j][i] = support;
      const std::int32_t q16 =
          posthoc_pair_cohesion_q16(traces, trace_count, i, j, work);
      matrix->cohesion[i][j] = q16;
      matrix->cohesion[j][i] = q16;
    }
}

inline std::uint32_t find_root(std::uint32_t* parent, std::uint32_t node) {
  while (parent[node] != node) node = parent[node];
  return node;
}

inline std::uint64_t fold_module_identity(
    const DirectDiscoveredModule& module) {
  std::uint64_t identity = 0x706f7374686f6331ull;
  for (std::uint32_t m = 0u; m < module.member_count; ++m) {
    identity ^= module.members[m] + 0x9e3779b97f4a7c15ull +
                (identity << 6) + (identity >> 2);
  }
  return identity == 0u ? 1u : identity | (1ull << 63);
}

}  // namespace posthoc_detail

// Post-hoc module + connectome derivation from measured traces alone.
// Deterministic: fixed iteration order, integer arithmetic, no randomness.
inline bool discover_posthoc_modules(
    const DirectActivationTrace* traces, std::uint32_t trace_count,
    std::uint32_t node_count, std::int32_t min_cohesion_q16,
    DirectPosthocDiscoveryReport* report) {
  if (report == nullptr || traces == nullptr || trace_count == 0u ||
      trace_count > kDiscoveryMaxTraces || node_count < 2u ||
      node_count > kDiscoveryMaxNodes)
    return false;
  *report = DirectPosthocDiscoveryReport{};

  posthoc_detail::PairCohesion matrix{};
  posthoc_detail::measure_all_pairs(traces, trace_count, node_count, &matrix,
                                    &report->work_units);
  report->samples_folded =
      static_cast<std::uint64_t>(trace_count) *
      (static_cast<std::uint64_t>(node_count) * (node_count - 1u)) / 2u;

  // Greedy union-find over qualifying pairs in fixed lexicographic order.
  // A node whose series never varies across windows carries no covariance
  // evidence; flat nodes cannot join any pair (flat-control refusal).
  std::uint32_t parent[kDiscoveryMaxNodes];
  bool varying[kDiscoveryMaxNodes];
  for (std::uint32_t i = 0u; i < node_count; ++i) {
    parent[i] = i;
    std::uint32_t lo = 0xffffffffu, hi = 0u;
    for (std::uint32_t t = 0u; t < trace_count; ++t) {
      const std::uint32_t v = traces[t].visits[i];
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
    varying[i] = hi != lo && hi != 0u;
  }
  for (std::uint32_t i = 0u; i + 1u < node_count; ++i)
    for (std::uint32_t j = i + 1u; j < node_count; ++j) {
      ++report->work_units;
      if (!varying[i] || !varying[j]) continue;
      if (matrix.support[i][j] < kDiscoveryMinPairSupport) continue;
      if (matrix.cohesion[i][j] < min_cohesion_q16) continue;
      const std::uint32_t ri = posthoc_detail::find_root(parent, i);
      const std::uint32_t rj = posthoc_detail::find_root(parent, j);
      if (ri != rj) parent[rj] = ri;
    }

  // Compact roots into bounded modules in first-appearance order; singleton
  // components carry no functional-module evidence and are not emitted.
  {
    std::uint32_t size[kDiscoveryMaxNodes]{};
    for (std::uint32_t i = 0u; i < node_count; ++i)
      ++size[posthoc_detail::find_root(parent, i)];
    std::uint32_t root_slot[kDiscoveryMaxNodes];
    for (std::uint32_t i = 0u; i < node_count; ++i) root_slot[i] = 0xffffffffu;
    for (std::uint32_t i = 0u; i < node_count; ++i) {
      const std::uint32_t root = posthoc_detail::find_root(parent, i);
      if (size[root] < 2u) continue;
      if (root_slot[root] == 0xffffffffu) {
        if (report->module_count >= kDiscoveryMaxModules) return false;
        root_slot[root] = report->module_count++;
        report->modules[root_slot[root]].member_count = 0u;
      }
      DirectDiscoveredModule& target =
          report->modules[root_slot[root]];
      if (target.member_count >= kDiscoveryMaxNodes) return false;
      target.members[target.member_count++] = i;
    }
  }
  for (std::uint32_t m = 0u; m < report->module_count; ++m) {
    DirectDiscoveredModule& module = report->modules[m];
    std::uint64_t pairs = 0u, sum = 0u;
    for (std::uint32_t a = 0u; a < module.member_count; ++a)
      for (std::uint32_t b = a + 1u; b < module.member_count; ++b) {
        sum += static_cast<std::uint64_t>(
            matrix.cohesion[module.members[a]][module.members[b]]);
        ++pairs;
        ++report->work_units;
      }
    module.internal_cohesion_q16 =
        pairs != 0u ? static_cast<std::int32_t>(sum / pairs)
                    : static_cast<std::int32_t>(1 << 16);
    module.module_identity = posthoc_detail::fold_module_identity(module);
  }

  // Inter-module connectome edges from measured cross-boundary coactivation.
  for (std::uint32_t a = 0u; a + 1u < report->module_count; ++a)
    for (std::uint32_t b = a + 1u; b < report->module_count; ++b) {
      std::uint64_t pairs = 0u, sum = 0u;
      const DirectDiscoveredModule& ma = report->modules[a];
      const DirectDiscoveredModule& mb = report->modules[b];
      for (std::uint32_t x = 0u; x < ma.member_count; ++x)
        for (std::uint32_t y = 0u; y < mb.member_count; ++y) {
          const std::int32_t q16 =
              matrix.cohesion[ma.members[x]][mb.members[y]];
          if (q16 > 0) {
            sum += static_cast<std::uint64_t>(q16);
            ++pairs;
          }
          ++report->work_units;
        }
      if (pairs == 0u) continue;
      if (report->edge_count >= kDiscoveryMaxModules) return false;
      DirectModuleEdge& edge = report->edges[report->edge_count++];
      edge.module_a = static_cast<std::uint16_t>(a);
      edge.module_b = static_cast<std::uint16_t>(b);
      edge.coupled_pairs = static_cast<std::uint32_t>(pairs);
      edge.mean_coactivation_q16 =
          static_cast<std::int32_t>(sum / pairs);
    }
  return true;
}

// Time-shuffled null: independently Fisher-Yates-permute each node's time
// column (fixed deterministic LCG), destroying cross-node alignment while
// preserving every marginal series exactly.
inline bool discover_control_time_shuffled(
    const DirectActivationTrace* traces, std::uint32_t trace_count,
    std::uint32_t node_count, std::int32_t min_cohesion_q16,
    std::uint64_t null_seed, DirectPosthocDiscoveryReport* report) {
  if (traces == nullptr || trace_count < 2u ||
      trace_count > kDiscoveryMaxTraces)
    return false;
  DirectActivationTrace shuffled[kDiscoveryMaxTraces];
  for (std::uint32_t t = 0u; t < trace_count; ++t) shuffled[t] = traces[t];
  for (std::uint32_t n = 0u; n < node_count && n < kDiscoveryMaxNodes; ++n) {
    std::uint32_t order[kDiscoveryMaxTraces];
    for (std::uint32_t t = 0u; t < trace_count; ++t) order[t] = t;
    std::uint64_t state =
        null_seed + 0x9e37ull * static_cast<std::uint64_t>(n);
    auto next = [&state]() {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      return static_cast<std::uint32_t>(state >> 33);
    };
    for (std::uint32_t t = trace_count - 1u; t > 0u; --t) {
      const std::uint32_t j = next() % (t + 1u);
      const std::uint32_t tmp = order[t];
      order[t] = order[j];
      order[j] = tmp;
    }
    std::uint32_t column[kDiscoveryMaxTraces];
    for (std::uint32_t t = 0u; t < trace_count; ++t)
      column[order[t]] = traces[t].visits[n];
    for (std::uint32_t t = 0u; t < trace_count; ++t)
      shuffled[t].visits[n] = column[t];
  }
  return discover_posthoc_modules(shuffled, trace_count, node_count,
                                  min_cohesion_q16, report);
}

}  // namespace substrate::direct_adult_core

#endif
