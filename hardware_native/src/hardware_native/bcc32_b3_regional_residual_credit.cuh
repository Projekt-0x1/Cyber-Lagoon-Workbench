#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_b3_route_eligibility_credit.cuh"

namespace bcc32_b3_regional_residual_credit {

namespace b3 = bcc32_b3_route_eligibility_credit;
namespace graph = bcc32_b1_resident_sparse_recurrent_graph;
using substrate::bcc32::SiteWord;

constexpr std::uint32_t kMatchedResidual = 3u;

struct RegionalJournal {
  graph::B2MutationEscrow route{};
  std::uint32_t edge_index = graph::kInvalidEdge;
  SiteWord eligibility = 0u;
  std::uint32_t armed_tick = b3::kNoTick;
  SiteWord residual = 0u;
  std::uint32_t residual_sign = 0u;
  std::uint32_t committed = 0u;
};

struct RegionalReceipt {
  std::uint64_t state_before_hash = 0ull;
  std::uint64_t state_after_hash = 0ull;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t selected_edge = graph::kInvalidEdge;
  std::uint32_t selected_region = graph::kInvalidEdge;
  std::uint32_t residual_sign = 0u;
  std::uint32_t armed_count = 0u;
  std::uint32_t committed = 0u;
  std::uint32_t matched = 0u;
  std::uint32_t expired = 0u;
};

struct DeviceRegionalCreditView {
  graph::DeviceGraphView graph{};
  b3::RouteEligibility* eligibility = nullptr;
  SiteWord* free_trace_bank = nullptr;
  SiteWord* positive_regions = nullptr;
  SiteWord* negative_regions = nullptr;
  SiteWord* matched_regions = nullptr;
  const SiteWord* source_activity = nullptr;
  const SiteWord* target_activity = nullptr;
  std::uint32_t source_activity_count = 0u;
  std::uint32_t target_activity_count = 0u;
  std::uint32_t region_count = 0u;
  std::uint32_t expiry_ticks = 0u;
  RegionalJournal* journal = nullptr;
};

static __global__ void xor_regional_contact_kernel(
    SiteWord* regions, std::uint32_t region_count, std::uint32_t region,
    SiteWord quantum) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || regions == nullptr ||
      region >= region_count)
    return;
  atomicXor(regions + region, quantum);
}

__device__ inline std::uint32_t matter_bits(
    const DeviceRegionalCreditView& view) {
  std::uint32_t total = graph::graph_matter_bits(view.graph);
  for (std::uint32_t edge = 0u; edge < view.graph.edge_count; ++edge)
    total += __popc(view.eligibility[edge].word);
  if (view.free_trace_bank != nullptr) total += __popc(*view.free_trace_bank);
  for (std::uint32_t region = 0u; region < view.region_count; ++region) {
    total += __popc(view.positive_regions[region]);
    total += __popc(view.negative_regions[region]);
    total += __popc(view.matched_regions[region]);
  }
  if (view.journal != nullptr && view.journal->committed != 0u) {
    total += __popc(view.journal->eligibility);
    total += __popc(view.journal->residual);
  }
  return total;
}

__device__ inline std::uint64_t state_hash(
    const DeviceRegionalCreditView& view) {
  std::uint64_t hash = graph::hash_graph_state(view.graph);
  for (std::uint32_t edge = 0u; edge < view.graph.edge_count; ++edge) {
    hash = graph::mix_word(hash, view.eligibility[edge].word,
                           0xa00u + edge * 2u);
    hash = graph::mix_word(hash, view.eligibility[edge].armed_tick,
                           0xa01u + edge * 2u);
  }
  hash = graph::mix_word(
      hash, view.free_trace_bank == nullptr ? 0u : *view.free_trace_bank,
      0xb00u);
  for (std::uint32_t region = 0u; region < view.region_count; ++region) {
    hash = graph::mix_word(hash, view.positive_regions[region],
                           0xc00u + region * 3u);
    hash = graph::mix_word(hash, view.negative_regions[region],
                           0xc01u + region * 3u);
    hash = graph::mix_word(hash, view.matched_regions[region],
                           0xc02u + region * 3u);
  }
  if (view.journal != nullptr) {
    hash = graph::mix_word(hash, view.journal->edge_index, 0xd00u);
    hash = graph::mix_word(hash, view.journal->eligibility, 0xd01u);
    hash = graph::mix_word(hash, view.journal->armed_tick, 0xd02u);
    hash = graph::mix_word(hash, view.journal->residual, 0xd03u);
    hash = graph::mix_word(hash, view.journal->residual_sign, 0xd04u);
    hash = graph::mix_word(hash, view.journal->committed, 0xd05u);
  }
  return hash;
}

__device__ inline bool locally_coactive(
    const DeviceRegionalCreditView& view, const graph::EdgeRecord& edge) {
  return view.source_activity != nullptr && view.target_activity != nullptr &&
         edge.source < view.source_activity_count &&
         edge.target < view.target_activity_count &&
         view.source_activity[edge.source] != 0u &&
         view.target_activity[edge.target] != 0u;
}

// Every locally coactive route earns its own finite trace quantum. No route
// index enters from the host or from the later residual.
static __global__ void arm_regional_eligibility_kernel(
    DeviceRegionalCreditView view, std::uint32_t tick,
    RegionalReceipt* receipt) {
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= view.graph.edge_count || view.eligibility == nullptr)
    return;
  const graph::EdgeRecord edge = view.graph.edges[edge_index];
  if (edge.source == graph::kInvalidEdge ||
      edge.target == graph::kInvalidEdge || !locally_coactive(view, edge) ||
      view.eligibility[edge_index].word != 0u)
    return;
  const SiteWord quantum = b3::take_trace_quantum(view.free_trace_bank);
  if (quantum == 0u) return;
  view.eligibility[edge_index].word = quantum;
  view.eligibility[edge_index].armed_tick = tick;
  if (receipt != nullptr) atomicAdd(&receipt->armed_count, 1u);
}

static __global__ void expire_regional_eligibility_kernel(
    DeviceRegionalCreditView view, std::uint32_t tick,
    RegionalReceipt* receipt) {
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= view.graph.edge_count || view.eligibility == nullptr)
    return;
  b3::RouteEligibility& trace = view.eligibility[edge_index];
  if (trace.word == 0u || tick - trace.armed_tick <= view.expiry_ticks) return;
  atomicOr(view.free_trace_bank, trace.word);
  trace = b3::RouteEligibility{};
  if (receipt != nullptr) atomicAdd(&receipt->expired, 1u);
}

__device__ inline SiteWord take_region_quantum(SiteWord* rail) {
  return b3::take_trace_quantum(rail);
}

static __global__ void commit_regional_residual_kernel(
    DeviceRegionalCreditView view, RegionalReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  RegionalReceipt local{};
  local.state_before_hash = state_hash(view);
  local.matter_before_bits = matter_bits(view);
  if (view.journal == nullptr || view.graph.development_escrow == nullptr ||
      view.graph.development_escrow->active != 0u ||
      view.journal->committed != 0u || view.graph.source_count <= 1u ||
      view.graph.target_count <= 1u) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    if (receipt != nullptr) *receipt = local;
    return;
  }

  for (std::uint32_t edge_index = 0u; edge_index < view.graph.edge_count;
       ++edge_index) {
    b3::RouteEligibility& trace = view.eligibility[edge_index];
    const graph::EdgeRecord old_edge = view.graph.edges[edge_index];
    if (trace.word == 0u || old_edge.source >= view.region_count ||
        old_edge.source == graph::kInvalidEdge ||
        old_edge.target == graph::kInvalidEdge)
      continue;
    const std::uint32_t region = old_edge.source;
    const SiteWord positive = view.positive_regions[region];
    const SiteWord negative = view.negative_regions[region];
    const SiteWord matched = view.matched_regions[region];
    const std::uint32_t active_rails =
        (positive != 0u ? 1u : 0u) + (negative != 0u ? 1u : 0u) +
        (matched != 0u ? 1u : 0u);
    if (active_rails != 1u) continue;
    if (matched != 0u) {
      local.selected_edge = edge_index;
      local.selected_region = region;
      local.residual_sign = kMatchedResidual;
      local.matched = 1u;
      break;
    }
    if (view.graph.lesion_escrow[old_edge.positive_slot] != 0u ||
        view.graph.lesion_escrow[old_edge.negative_slot] != 0u)
      continue;

    const std::uint32_t sign =
        positive != 0u ? b3::kPositiveResidual : b3::kNegativeResidual;
    SiteWord* rail = positive != 0u ? view.positive_regions + region
                                    : view.negative_regions + region;
    const SiteWord residual = take_region_quantum(rail);
    if (residual == 0u) continue;

    graph::B2MutationEscrow escrow{};
    escrow.old_edge = old_edge;
    escrow.live_edge_index = edge_index;
    escrow.positive_word =
        view.graph.route_words[old_edge.positive_slot];
    escrow.negative_word =
        view.graph.route_words[old_edge.negative_slot];
    escrow.active = 1u;
    *view.graph.development_escrow = escrow;
    view.graph.route_words[old_edge.positive_slot] = 0u;
    view.graph.route_words[old_edge.negative_slot] = 0u;

    const std::uint32_t sign_salt =
        sign == b3::kPositiveResidual ? 0x27d4eb2du : 0xa511e9b3u;
    const std::uint32_t seed =
        graph::b2_edge_key(old_edge, edge_index) ^ trace.word ^ sign_salt;
    graph::EdgeRecord changed = old_edge;
    changed.source =
        (old_edge.source + 1u + seed % (view.graph.source_count - 1u)) %
        view.graph.source_count;
    changed.target =
        (old_edge.target + 1u +
         ((seed >> 8u) % (view.graph.target_count - 1u))) %
        view.graph.target_count;
    view.graph.edges[edge_index] = changed;

    view.journal->route = escrow;
    view.journal->edge_index = edge_index;
    view.journal->eligibility = trace.word;
    view.journal->armed_tick = trace.armed_tick;
    view.journal->residual = residual;
    view.journal->residual_sign = sign;
    view.journal->committed = 1u;
    trace = b3::RouteEligibility{};
    local.selected_edge = edge_index;
    local.selected_region = region;
    local.residual_sign = sign;
    local.committed = 1u;
    break;
  }
  local.state_after_hash = state_hash(view);
  local.matter_after_bits = matter_bits(view);
  if (receipt != nullptr) *receipt = local;
}

static __global__ void reverse_regional_residual_kernel(
    DeviceRegionalCreditView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.journal == nullptr ||
      view.journal->committed == 0u ||
      view.journal->edge_index >= view.graph.edge_count)
    return;
  const RegionalJournal journal = *view.journal;
  view.graph.edges[journal.edge_index] = journal.route.old_edge;
  view.graph.route_words[journal.route.old_edge.positive_slot] =
      journal.route.positive_word;
  view.graph.route_words[journal.route.old_edge.negative_slot] =
      journal.route.negative_word;
  *view.graph.development_escrow = graph::B2MutationEscrow{};
  view.eligibility[journal.edge_index] = {
      journal.eligibility, journal.armed_tick};
  const std::uint32_t region = journal.route.old_edge.source;
  if (journal.residual_sign == b3::kPositiveResidual)
    atomicOr(view.positive_regions + region, journal.residual);
  else if (journal.residual_sign == b3::kNegativeResidual)
    atomicOr(view.negative_regions + region, journal.residual);
  *view.journal = RegionalJournal{};
}

static __global__ void reverse_regional_arm_kernel(
    DeviceRegionalCreditView view) {
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= view.graph.edge_count || view.eligibility == nullptr)
    return;
  b3::RouteEligibility& trace = view.eligibility[edge_index];
  if (trace.word == 0u) return;
  atomicOr(view.free_trace_bank, trace.word);
  trace = b3::RouteEligibility{};
}

static __global__ void state_hash_kernel(DeviceRegionalCreditView view,
                                         std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *output = state_hash(view);
}

static __global__ void matter_bits_kernel(DeviceRegionalCreditView view,
                                          std::uint32_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *output = matter_bits(view);
}

}  // namespace bcc32_b3_regional_residual_credit
