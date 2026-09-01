#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_b1_resident_sparse_recurrent_graph.cuh"

// B3 is deliberately a sidecar to the B2 graph ABI.  The graph remains the
// authority for route matter and projection; this file adds one finite,
// conserved eligibility transaction around an existing B2 mutation escrow.
namespace bcc32_b3_route_eligibility_credit {

namespace graph = bcc32_b1_resident_sparse_recurrent_graph;
using substrate::bcc32::SiteWord;

constexpr std::uint32_t kInvalidEdge = graph::kInvalidEdge;
constexpr std::uint32_t kNoTick = 0xffffffffu;
constexpr std::uint32_t kPositiveResidual = 1u;
constexpr std::uint32_t kNegativeResidual = 2u;

struct RouteEligibility {
  SiteWord word = 0u;
  std::uint32_t armed_tick = kNoTick;
};

// One B2 relocation can be verified at a time. The old route matter remains
// in graph.development_escrow; the trace and residual rails move into this
// journal only while a commit is active.
struct B3InverseJournal {
  std::uint32_t edge_index = kInvalidEdge;
  std::uint32_t armed_tick = kNoTick;
  SiteWord eligibility_escrow = 0u;
  SiteWord positive_residual_escrow = 0u;
  SiteWord negative_residual_escrow = 0u;
  SiteWord expiry_mask = 0u;
  std::uint32_t residual_sign = 0u;
  std::uint32_t armed = 0u;
  std::uint32_t committed = 0u;
  std::uint32_t expired = 0u;
};

struct B3CreditReceipt {
  std::uint64_t state_before_hash = 0ull;
  std::uint64_t state_after_hash = 0ull;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t selected_edge = kInvalidEdge;
  std::uint32_t residual_sign = 0u;
  std::uint32_t armed = 0u;
  std::uint32_t committed = 0u;
  std::uint32_t expired = 0u;
};

struct DeviceRouteEligibilityView {
  graph::DeviceGraphView graph{};
  RouteEligibility* eligibility = nullptr;
  SiteWord* free_trace_bank = nullptr;
  SiteWord* positive_residual = nullptr;
  SiteWord* negative_residual = nullptr;
  const SiteWord* source_activity = nullptr;
  const SiteWord* target_activity = nullptr;
  std::uint32_t source_activity_count = 0u;
  std::uint32_t target_activity_count = 0u;
  std::uint32_t expiry_ticks = 0u;
  B3InverseJournal* journal = nullptr;
};

__host__ __device__ inline std::uint32_t b3_route_key(
    const graph::EdgeRecord& edge, std::uint32_t edge_index,
    SiteWord eligibility_word) {
  return graph::b2_edge_key(edge, edge_index) ^ eligibility_word;
}

__device__ inline SiteWord take_trace_quantum(SiteWord* bank) {
  if (bank == nullptr) return 0u;
  SiteWord observed = *bank;
  while (observed != 0u) {
    const SiteWord quantum = observed & (0u - observed);
    const SiteWord desired = observed & ~quantum;
    const SiteWord prior = atomicCAS(bank, observed, desired);
    if (prior == observed) return quantum;
    observed = prior;
  }
  return 0u;
}

__device__ inline bool edge_is_locally_coactive(
    const DeviceRouteEligibilityView& view, const graph::EdgeRecord& edge) {
  return view.source_activity != nullptr && view.target_activity != nullptr &&
         edge.source < view.source_activity_count &&
         edge.target < view.target_activity_count &&
         view.source_activity[edge.source] != 0u &&
         view.target_activity[edge.target] != 0u;
}

__device__ inline std::uint32_t b3_matter_bits(
    const DeviceRouteEligibilityView& view) {
  std::uint32_t total = graph::graph_matter_bits(view.graph);
  for (std::uint32_t index = 0u; index < view.graph.edge_count; ++index)
    total += __popc(view.eligibility[index].word);
  if (view.free_trace_bank != nullptr) total += __popc(*view.free_trace_bank);
  if (view.positive_residual != nullptr) total += __popc(*view.positive_residual);
  if (view.negative_residual != nullptr) total += __popc(*view.negative_residual);
  if (view.journal != nullptr && view.journal->committed != 0u) {
    total += __popc(view.journal->eligibility_escrow);
    total += __popc(view.journal->positive_residual_escrow);
    total += __popc(view.journal->negative_residual_escrow);
  }
  return total;
}

__device__ inline std::uint64_t hash_b3_state(
    const DeviceRouteEligibilityView& view) {
  std::uint64_t hash = graph::hash_graph_state(view.graph);
  for (std::uint32_t index = 0u; index < view.graph.edge_count; ++index) {
    hash = graph::mix_word(hash, view.eligibility[index].word, 0x800u + index * 2u);
    hash = graph::mix_word(hash, view.eligibility[index].armed_tick,
                           0x801u + index * 2u);
  }
  hash = graph::mix_word(hash,
                         view.free_trace_bank == nullptr ? 0u : *view.free_trace_bank,
                         0x900u);
  hash = graph::mix_word(hash,
                         view.positive_residual == nullptr ? 0u : *view.positive_residual,
                         0x901u);
  hash = graph::mix_word(hash,
                         view.negative_residual == nullptr ? 0u : *view.negative_residual,
                         0x902u);
  if (view.journal != nullptr) {
    const B3InverseJournal journal = *view.journal;
    hash = graph::mix_word(hash, journal.edge_index, 0x910u);
    hash = graph::mix_word(hash, journal.armed_tick, 0x911u);
    hash = graph::mix_word(hash, journal.eligibility_escrow, 0x912u);
    hash = graph::mix_word(hash, journal.positive_residual_escrow, 0x913u);
    hash = graph::mix_word(hash, journal.negative_residual_escrow, 0x914u);
    hash = graph::mix_word(hash, journal.expiry_mask, 0x915u);
    hash = graph::mix_word(hash, journal.residual_sign, 0x916u);
    hash = graph::mix_word(hash, journal.armed, 0x917u);
    hash = graph::mix_word(hash, journal.committed, 0x918u);
    hash = graph::mix_word(hash, journal.expired, 0x919u);
  }
  return hash;
}

// Every edge independently tests only its endpoints. A successful arm moves a
// single represented trace quantum out of the shared free bank; no host route
// id is present in this path.
static __global__ void arm_route_eligibility_kernel(
    DeviceRouteEligibilityView view, std::uint32_t tick, B3CreditReceipt* receipt) {
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= view.graph.edge_count || view.eligibility == nullptr ||
      view.journal == nullptr)
    return;
  const graph::EdgeRecord edge = view.graph.edges[edge_index];
  if (edge.source == kInvalidEdge || edge.target == kInvalidEdge ||
      !edge_is_locally_coactive(view, edge) || view.eligibility[edge_index].word != 0u)
    return;
  if (atomicCAS(&view.journal->armed, 0u, 1u) != 0u) return;
  const SiteWord quantum = take_trace_quantum(view.free_trace_bank);
  if (quantum == 0u) {
    view.journal->armed = 0u;
    return;
  }
  view.eligibility[edge_index].word = quantum;
  view.eligibility[edge_index].armed_tick = tick;
  view.journal->edge_index = edge_index;
  view.journal->armed_tick = tick;
  if (receipt != nullptr) {
    receipt->selected_edge = edge_index;
    receipt->armed = 1u;
  }
}

// Expiry returns trace matter to the free bank. expiry_mask is inverse metadata
// only; it is never read by the forward commit rule.
static __global__ void expire_route_eligibility_kernel(
    DeviceRouteEligibilityView view, std::uint32_t tick, B3CreditReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.journal == nullptr ||
      view.journal->armed == 0u || view.journal->committed != 0u ||
      view.journal->expired != 0u || view.journal->edge_index >= view.graph.edge_count)
    return;
  const std::uint32_t edge_index = view.journal->edge_index;
  RouteEligibility& trace = view.eligibility[edge_index];
  if (trace.word == 0u || tick - trace.armed_tick <= view.expiry_ticks) return;
  const SiteWord released = trace.word;
  atomicOr(view.free_trace_bank, released);
  trace.word = 0u;
  trace.armed_tick = kNoTick;
  view.journal->expiry_mask = released;
  view.journal->expired = 1u;
  if (receipt != nullptr) {
    receipt->selected_edge = edge_index;
    receipt->expired = 1u;
  }
}

// The later residual is represented by exactly one sign rail. Its word must
// equal the local edge+trace key (or its complement on the negative rail), so
// an equally strong but ineligible route cannot be selected.
__device__ inline void commit_eligible_residual(
    DeviceRouteEligibilityView view, B3CreditReceipt* receipt) {
  B3CreditReceipt local{};
  local.state_before_hash = hash_b3_state(view);
  local.matter_before_bits = b3_matter_bits(view);
  if (view.journal == nullptr || view.graph.development_escrow == nullptr ||
      view.positive_residual == nullptr || view.negative_residual == nullptr ||
      view.journal->armed == 0u || view.journal->committed != 0u ||
      view.journal->expired != 0u || view.graph.development_escrow->active != 0u) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    if (receipt != nullptr) *receipt = local;
    return;
  }

  const SiteWord positive = *view.positive_residual;
  const SiteWord negative = *view.negative_residual;
  if ((positive == 0u) == (negative == 0u)) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    if (receipt != nullptr) *receipt = local;
    return;
  }
  const std::uint32_t sign = positive != 0u ? kPositiveResidual : kNegativeResidual;
  const SiteWord rail_word = positive != 0u ? positive : negative;
  const SiteWord required_key = sign == kPositiveResidual ? rail_word : ~rail_word;

  const std::uint32_t edge_index = view.journal->edge_index;
  if (edge_index >= view.graph.edge_count) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    if (receipt != nullptr) *receipt = local;
    return;
  }
  const graph::EdgeRecord old_edge = view.graph.edges[edge_index];
  RouteEligibility& trace = view.eligibility[edge_index];
  if (trace.word == 0u || old_edge.source == kInvalidEdge ||
      old_edge.target == kInvalidEdge ||
      b3_route_key(old_edge, edge_index, trace.word) != required_key ||
      view.graph.lesion_escrow[old_edge.positive_slot] != 0u ||
      view.graph.lesion_escrow[old_edge.negative_slot] != 0u) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    if (receipt != nullptr) *receipt = local;
    return;
  }

  const SiteWord old_positive = view.graph.route_words[old_edge.positive_slot];
  const SiteWord old_negative = view.graph.route_words[old_edge.negative_slot];
  graph::B2MutationEscrow escrow{};
  escrow.old_edge = old_edge;
  escrow.live_edge_index = edge_index;
  escrow.positive_word = old_positive;
  escrow.negative_word = old_negative;
  escrow.active = 1u;
  *view.graph.development_escrow = escrow;
  view.graph.route_words[old_edge.positive_slot] = 0u;
  view.graph.route_words[old_edge.negative_slot] = 0u;

  const std::uint32_t sign_salt =
      sign == kPositiveResidual ? 0u : 0xa511e9b3u;
  const std::uint32_t seed = graph::b2_edge_key(old_edge, edge_index) ^ trace.word ^
                             required_key ^ sign_salt;
  graph::EdgeRecord grown = old_edge;
  grown.source = (old_edge.source + 1u + seed % (view.graph.source_count - 1u)) %
                 view.graph.source_count;
  grown.target =
      (old_edge.target + 1u + ((seed >> 8u) % (view.graph.target_count - 1u))) %
      view.graph.target_count;
  view.graph.edges[edge_index] = grown;

  view.journal->eligibility_escrow = trace.word;
  trace.word = 0u;
  trace.armed_tick = kNoTick;
  view.journal->residual_sign = sign;
  if (sign == kPositiveResidual) {
    view.journal->positive_residual_escrow = positive;
    *view.positive_residual = 0u;
  } else {
    view.journal->negative_residual_escrow = negative;
    *view.negative_residual = 0u;
  }
  view.journal->committed = 1u;
  local.selected_edge = edge_index;
  local.residual_sign = sign;
  local.armed = 1u;
  local.committed = 1u;
  local.state_after_hash = hash_b3_state(view);
  local.matter_after_bits = b3_matter_bits(view);
  if (receipt != nullptr) *receipt = local;
}

static __global__ void commit_eligible_residual_kernel(
    DeviceRouteEligibilityView view, B3CreditReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  commit_eligible_residual(view, receipt);
}

__device__ inline void reverse_eligible_residual(
    DeviceRouteEligibilityView view) {
  if (view.journal == nullptr ||
      view.journal->committed == 0u || view.journal->edge_index >= view.graph.edge_count ||
      view.graph.development_escrow == nullptr)
    return;
  const graph::B2MutationEscrow escrow = *view.graph.development_escrow;
  if (escrow.active == 0u || escrow.live_edge_index != view.journal->edge_index) return;
  view.graph.route_words[escrow.old_edge.positive_slot] = escrow.positive_word;
  view.graph.route_words[escrow.old_edge.negative_slot] = escrow.negative_word;
  view.graph.edges[escrow.live_edge_index] = escrow.old_edge;
  *view.graph.development_escrow = graph::B2MutationEscrow{};
  RouteEligibility& trace = view.eligibility[view.journal->edge_index];
  trace.word = view.journal->eligibility_escrow;
  trace.armed_tick = view.journal->armed_tick;
  if (view.journal->residual_sign == kPositiveResidual)
    *view.positive_residual = view.journal->positive_residual_escrow;
  else if (view.journal->residual_sign == kNegativeResidual)
    *view.negative_residual = view.journal->negative_residual_escrow;
  view.journal->eligibility_escrow = 0u;
  view.journal->positive_residual_escrow = 0u;
  view.journal->negative_residual_escrow = 0u;
  view.journal->residual_sign = 0u;
  view.journal->committed = 0u;
}

static __global__ void reverse_eligible_residual_kernel(
    DeviceRouteEligibilityView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  reverse_eligible_residual(view);
}

static __global__ void reverse_eligibility_expiry_kernel(
    DeviceRouteEligibilityView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.journal == nullptr ||
      view.journal->expired == 0u || view.journal->edge_index >= view.graph.edge_count)
    return;
  const SiteWord released = view.journal->expiry_mask;
  *view.free_trace_bank &= ~released;
  RouteEligibility& trace = view.eligibility[view.journal->edge_index];
  trace.word = released;
  trace.armed_tick = view.journal->armed_tick;
  view.journal->expiry_mask = 0u;
  view.journal->expired = 0u;
}

static __global__ void reverse_route_eligibility_kernel(
    DeviceRouteEligibilityView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.journal == nullptr ||
      view.journal->armed == 0u || view.journal->committed != 0u ||
      view.journal->expired != 0u || view.journal->edge_index >= view.graph.edge_count)
    return;
  RouteEligibility& trace = view.eligibility[view.journal->edge_index];
  atomicOr(view.free_trace_bank, trace.word);
  trace = RouteEligibility{};
  *view.journal = B3InverseJournal{};
}

static __global__ void hash_b3_state_kernel(DeviceRouteEligibilityView view,
                                             std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *output = hash_b3_state(view);
}

static __global__ void b3_matter_bits_kernel(DeviceRouteEligibilityView view,
                                             std::uint32_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *output = b3_matter_bits(view);
}

}  // namespace bcc32_b3_route_eligibility_credit
