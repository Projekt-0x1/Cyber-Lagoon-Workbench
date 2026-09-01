#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_b3_route_eligibility_credit.cuh"

namespace bcc32_b3_four_contact_eligibility {

namespace b3 = bcc32_b3_route_eligibility_credit;
namespace graph = bcc32_b1_resident_sparse_recurrent_graph;
using substrate::bcc32::SiteWord;

constexpr std::uint32_t kEligibilityHorizon = 4u;
constexpr std::uint32_t kInvalidLane = 0xffffffffu;

struct FourContactScalars {
  std::uint32_t next_lane = 0u;
  std::uint32_t active_count = 0u;
  std::uint32_t committed_lane = kInvalidLane;
  std::uint32_t commit_count_before = 0u;
};

struct ExpiryJournal {
  b3::RouteEligibility previous_trace[kEligibilityHorizon]{};
  b3::B3InverseJournal previous_lane_journal[kEligibilityHorizon]{};
  FourContactScalars previous_scalars{};
  SiteWord free_trace_before = 0u;
  std::uint32_t changed_mask = 0u;
};

struct DeviceFourContactView {
  graph::DeviceGraphView graph{};
  b3::RouteEligibility* eligibility = nullptr;
  b3::B3InverseJournal* lane_journals = nullptr;
  FourContactScalars* scalars = nullptr;
  SiteWord* free_trace_bank = nullptr;
  SiteWord* positive_residual = nullptr;
  SiteWord* negative_residual = nullptr;
  const SiteWord* source_activity = nullptr;
  const SiteWord* target_activity = nullptr;
  std::uint32_t source_activity_count = 0u;
  std::uint32_t target_activity_count = 0u;
  std::uint32_t expiry_ticks = kEligibilityHorizon;
};

__host__ __device__ inline b3::DeviceRouteEligibilityView lane_view(
    const DeviceFourContactView& view, std::uint32_t lane) {
  return {
      view.graph,
      view.eligibility + lane * view.graph.edge_count,
      view.free_trace_bank,
      view.positive_residual,
      view.negative_residual,
      view.source_activity,
      view.target_activity,
      view.source_activity_count,
      view.target_activity_count,
      view.expiry_ticks,
      view.lane_journals + lane,
  };
}

__device__ inline std::uint32_t four_contact_matter_bits(
    const DeviceFourContactView& view) {
  std::uint32_t total = graph::graph_matter_bits(view.graph);
  for (std::uint32_t lane = 0u; lane < kEligibilityHorizon; ++lane) {
    const b3::DeviceRouteEligibilityView local = lane_view(view, lane);
    for (std::uint32_t edge = 0u; edge < view.graph.edge_count; ++edge)
      total += __popc(local.eligibility[edge].word);
    const b3::B3InverseJournal journal = local.journal[0];
    if (journal.committed != 0u) {
      total += __popc(journal.eligibility_escrow);
      total += __popc(journal.positive_residual_escrow);
      total += __popc(journal.negative_residual_escrow);
    }
  }
  if (view.free_trace_bank != nullptr) total += __popc(*view.free_trace_bank);
  if (view.positive_residual != nullptr) total += __popc(*view.positive_residual);
  if (view.negative_residual != nullptr) total += __popc(*view.negative_residual);
  return total;
}

__device__ inline std::uint64_t hash_four_contact_state(
    const DeviceFourContactView& view) {
  std::uint64_t hash = graph::hash_graph_state(view.graph);
  for (std::uint32_t lane = 0u; lane < kEligibilityHorizon; ++lane) {
    const b3::DeviceRouteEligibilityView local = lane_view(view, lane);
    for (std::uint32_t edge = 0u; edge < view.graph.edge_count; ++edge) {
      const b3::RouteEligibility trace = local.eligibility[edge];
      const std::uint32_t salt = 0xa00u + lane * 0x100u + edge * 2u;
      hash = graph::mix_word(hash, trace.word, salt);
      hash = graph::mix_word(hash, trace.armed_tick, salt + 1u);
    }
    const b3::B3InverseJournal journal = local.journal[0];
    hash = graph::mix_word(hash, journal.edge_index, 0xe00u + lane * 16u);
    hash = graph::mix_word(hash, journal.armed_tick, 0xe01u + lane * 16u);
    hash = graph::mix_word(hash, journal.eligibility_escrow,
                           0xe02u + lane * 16u);
    hash = graph::mix_word(hash, journal.positive_residual_escrow,
                           0xe03u + lane * 16u);
    hash = graph::mix_word(hash, journal.negative_residual_escrow,
                           0xe04u + lane * 16u);
    hash = graph::mix_word(hash, journal.expiry_mask,
                           0xe05u + lane * 16u);
    hash = graph::mix_word(hash, journal.residual_sign,
                           0xe06u + lane * 16u);
    hash = graph::mix_word(hash, journal.armed, 0xe07u + lane * 16u);
    hash = graph::mix_word(hash, journal.committed, 0xe08u + lane * 16u);
    hash = graph::mix_word(hash, journal.expired, 0xe09u + lane * 16u);
  }
  const FourContactScalars scalars =
      view.scalars == nullptr ? FourContactScalars{} : *view.scalars;
  hash = graph::mix_word(hash, scalars.next_lane, 0xf00u);
  hash = graph::mix_word(hash, scalars.active_count, 0xf01u);
  hash = graph::mix_word(hash, scalars.committed_lane, 0xf02u);
  hash = graph::mix_word(hash, scalars.commit_count_before, 0xf03u);
  hash = graph::mix_word(
      hash, view.free_trace_bank == nullptr ? 0u : *view.free_trace_bank, 0xf04u);
  hash = graph::mix_word(
      hash, view.positive_residual == nullptr ? 0u : *view.positive_residual,
      0xf05u);
  hash = graph::mix_word(
      hash, view.negative_residual == nullptr ? 0u : *view.negative_residual,
      0xf06u);
  return hash;
}

static __global__ void arm_four_contact_eligibility_kernel(
    DeviceFourContactView view, std::uint32_t tick,
    b3::B3CreditReceipt* receipt) {
  if (blockIdx.x != 0u) return;
  if (threadIdx.x == 0u && receipt != nullptr) *receipt = b3::B3CreditReceipt{};
  __syncthreads();
  if (view.scalars == nullptr || view.eligibility == nullptr ||
      view.lane_journals == nullptr ||
      view.scalars->active_count >= kEligibilityHorizon)
    return;
  const std::uint32_t lane = view.scalars->next_lane;
  if (lane >= kEligibilityHorizon) return;
  b3::DeviceRouteEligibilityView local = lane_view(view, lane);
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= view.graph.edge_count) return;
  const graph::EdgeRecord edge = view.graph.edges[edge_index];
  if (edge.source == b3::kInvalidEdge || edge.target == b3::kInvalidEdge ||
      !b3::edge_is_locally_coactive(local, edge) ||
      local.eligibility[edge_index].word != 0u)
    return;
  if (atomicCAS(&local.journal->armed, 0u, 1u) != 0u) return;
  const SiteWord quantum = b3::take_trace_quantum(view.free_trace_bank);
  if (quantum == 0u) {
    local.journal->armed = 0u;
    return;
  }
  local.eligibility[edge_index].word = quantum;
  local.eligibility[edge_index].armed_tick = tick;
  local.journal->edge_index = edge_index;
  local.journal->armed_tick = tick;
  view.scalars->next_lane = (lane + 1u) % kEligibilityHorizon;
  ++view.scalars->active_count;
  if (receipt != nullptr) {
    receipt->selected_edge = edge_index;
    receipt->armed = 1u;
  }
}

static __global__ void expire_four_contact_eligibility_kernel(
    DeviceFourContactView view, std::uint32_t tick, ExpiryJournal* history,
    b3::B3CreditReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.scalars == nullptr) return;
  if (receipt != nullptr) *receipt = b3::B3CreditReceipt{};
  if (history == nullptr || history->changed_mask != 0u) return;
  *history = ExpiryJournal{};
  history->previous_scalars = *view.scalars;
  history->free_trace_before =
      view.free_trace_bank == nullptr ? 0u : *view.free_trace_bank;
  for (std::uint32_t lane = 0u; lane < kEligibilityHorizon; ++lane) {
    b3::DeviceRouteEligibilityView local = lane_view(view, lane);
    b3::B3InverseJournal& journal = local.journal[0];
    if (journal.armed == 0u || journal.committed != 0u ||
        journal.expired != 0u || journal.edge_index >= view.graph.edge_count)
      continue;
    b3::RouteEligibility& trace = local.eligibility[journal.edge_index];
    if (trace.word == 0u || tick - trace.armed_tick <= view.expiry_ticks) continue;
    history->previous_trace[lane] = trace;
    history->previous_lane_journal[lane] = journal;
    history->changed_mask |= 1u << lane;
    const SiteWord released = trace.word;
    atomicOr(view.free_trace_bank, released);
    trace = b3::RouteEligibility{};
    journal = b3::B3InverseJournal{};
    if (view.scalars->active_count != 0u) --view.scalars->active_count;
    if (receipt != nullptr) {
      receipt->selected_edge = journal.edge_index;
      receipt->expired = 1u;
    }
  }
  if (history->changed_mask == 0u) *history = ExpiryJournal{};
}

static __global__ void commit_four_contact_residual_kernel(
    DeviceFourContactView view, b3::B3CreditReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.scalars == nullptr) return;
  b3::B3CreditReceipt result{};
  for (std::uint32_t lane = 0u; lane < kEligibilityHorizon; ++lane) {
    b3::B3CreditReceipt local{};
    b3::commit_eligible_residual(lane_view(view, lane), &local);
    if (local.committed == 0u) continue;
    view.scalars->committed_lane = lane;
    view.scalars->commit_count_before = view.scalars->active_count;
    if (view.scalars->active_count != 0u) --view.scalars->active_count;
    result = local;
    break;
  }
  if (receipt != nullptr) *receipt = result;
}

static __global__ void reverse_four_contact_commit_kernel(
    DeviceFourContactView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.scalars == nullptr ||
      view.scalars->committed_lane >= kEligibilityHorizon)
    return;
  const std::uint32_t lane = view.scalars->committed_lane;
  b3::reverse_eligible_residual(lane_view(view, lane));
  view.scalars->active_count = view.scalars->commit_count_before;
  view.scalars->committed_lane = kInvalidLane;
  view.scalars->commit_count_before = 0u;
}

static __global__ void reverse_four_contact_expiry_kernel(
    DeviceFourContactView view, ExpiryJournal* history) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.scalars == nullptr ||
      history == nullptr)
    return;
  *view.scalars = history->previous_scalars;
  if (view.free_trace_bank != nullptr)
    *view.free_trace_bank = history->free_trace_before;
  for (std::uint32_t lane = 0u; lane < kEligibilityHorizon; ++lane) {
    if ((history->changed_mask & (1u << lane)) == 0u) continue;
    b3::DeviceRouteEligibilityView local = lane_view(view, lane);
    const std::uint32_t edge = history->previous_lane_journal[lane].edge_index;
    if (edge < view.graph.edge_count)
      local.eligibility[edge] = history->previous_trace[lane];
    local.journal[0] = history->previous_lane_journal[lane];
  }
  *history = ExpiryJournal{};
}

// Arm state is self-inverting from the checkpointed lane and scalar state; no
// host-side per-contact history is required.
static __global__ void reverse_latest_four_contact_arm_kernel(
    DeviceFourContactView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.scalars == nullptr ||
      view.scalars->active_count == 0u)
    return;
  const std::uint32_t lane =
      (view.scalars->next_lane + kEligibilityHorizon - 1u) %
      kEligibilityHorizon;
  b3::DeviceRouteEligibilityView local = lane_view(view, lane);
  b3::B3InverseJournal& journal = local.journal[0];
  if (journal.armed == 0u || journal.committed != 0u ||
      journal.expired != 0u || journal.edge_index >= view.graph.edge_count)
    return;
  b3::RouteEligibility& trace = local.eligibility[journal.edge_index];
  if (trace.word == 0u) return;
  atomicOr(view.free_trace_bank, trace.word);
  trace = b3::RouteEligibility{};
  journal = b3::B3InverseJournal{};
  view.scalars->next_lane = lane;
  --view.scalars->active_count;
}

static __global__ void four_contact_matter_bits_kernel(
    DeviceFourContactView view, std::uint32_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || output == nullptr) return;
  *output = four_contact_matter_bits(view);
}

static __global__ void hash_four_contact_state_kernel(
    DeviceFourContactView view, std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || output == nullptr) return;
  *output = hash_four_contact_state(view);
}

}  // namespace bcc32_b3_four_contact_eligibility
