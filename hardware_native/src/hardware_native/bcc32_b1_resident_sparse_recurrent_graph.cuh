#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

namespace bcc32_b1_resident_sparse_recurrent_graph {

using substrate::bcc32::SiteWord;

constexpr std::uint32_t kInvalidEdge = 0xffffffffu;

struct EdgeRecord {
  std::uint32_t source = 0u;
  std::uint32_t target = 0u;
  std::uint32_t positive_slot = 0u;
  std::uint32_t negative_slot = 0u;
};

struct B2MutationEscrow {
  EdgeRecord old_edge{};
  std::uint32_t live_edge_index = kInvalidEdge;
  SiteWord positive_word = 0u;
  SiteWord negative_word = 0u;
  std::uint32_t active = 0u;
};

struct B2MutationReceipt {
  std::uint64_t state_before_hash = 0ull;
  std::uint64_t state_after_hash = 0ull;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t selected_edge = kInvalidEdge;
  std::uint32_t new_source = kInvalidEdge;
  std::uint32_t new_target = kInvalidEdge;
  std::uint32_t moved_quanta = 0u;
  std::uint32_t grew = 0u;
  std::uint32_t reversed = 0u;
};

struct DeviceGraphView {
  EdgeRecord* edges = nullptr;
  std::uint32_t edge_count = 0u;
  SiteWord* route_words = nullptr;
  SiteWord* lesion_escrow = nullptr;
  float quantum_scale = 0.0f;
  std::uint32_t route_word_count = 0u;
  std::uint32_t source_count = 0u;
  std::uint32_t target_count = 0u;
  B2MutationEscrow* development_escrow = nullptr;
};

struct GraphReceipt {
  std::uint64_t matter_before = 0ull;
  std::uint64_t matter_after = 0ull;
  std::uint64_t escrow_after = 0ull;
  std::uint32_t moved_quanta = 0u;
};

__host__ __device__ inline std::uint32_t b2_edge_key(
    const EdgeRecord& edge, std::uint32_t edge_index) {
  std::uint32_t key = 0x9e3779b9u;
  key ^= edge.source * 0x85ebca6bu;
  key = (key << 13u) | (key >> 19u);
  key ^= edge.target * 0xc2b2ae35u;
  key = (key << 11u) | (key >> 21u);
  key ^= edge.positive_slot * 0x27d4eb2du;
  key ^= edge.negative_slot * 0x165667b1u;
  key ^= edge_index * 0x7feb352du;
  key ^= key >> 16u;
  key *= 0x7feb352du;
  key ^= key >> 15u;
  return key;
}

__device__ inline std::uint32_t graph_route_word_count(
    const DeviceGraphView graph) {
  return graph.route_word_count != 0u ? graph.route_word_count
                                     : graph.edge_count * 2u;
}

__device__ inline SiteWord positive_matter(
    const DeviceGraphView graph, std::uint32_t edge_index,
    const EdgeRecord& edge) {
  if (graph.development_escrow != nullptr &&
      graph.development_escrow->active != 0u &&
      graph.development_escrow->live_edge_index == edge_index)
    return graph.development_escrow->positive_word;
  return graph.route_words[edge.positive_slot];
}

__device__ inline SiteWord negative_matter(
    const DeviceGraphView graph, std::uint32_t edge_index,
    const EdgeRecord& edge) {
  if (graph.development_escrow != nullptr &&
      graph.development_escrow->active != 0u &&
      graph.development_escrow->live_edge_index == edge_index)
    return graph.development_escrow->negative_word;
  return graph.route_words[edge.negative_slot];
}

__device__ inline std::int32_t signed_support(
    const DeviceGraphView graph, std::uint32_t edge_index,
    const EdgeRecord& edge) {
  return static_cast<std::int32_t>(
             __popc(positive_matter(graph, edge_index, edge))) -
         static_cast<std::int32_t>(
             __popc(negative_matter(graph, edge_index, edge)));
}

__device__ inline std::uint64_t mix_word(std::uint64_t hash,
                                         std::uint32_t word,
                                         std::uint32_t index) {
  hash ^= static_cast<std::uint64_t>(word) +
          0x9e3779b97f4a7c15ull +
          (static_cast<std::uint64_t>(index) << 6u) +
          (static_cast<std::uint64_t>(index) >> 2u);
  hash *= 1099511628211ull;
  return hash;
}

// Each edge is a long-range route. Spatial adjacency is not consulted.
// Positive and negative SiteWord populations are the sole edge-strength
// authority; the float scale is a disclosed B0 migration scaffold.
static __global__ void project_recurrent_routes_kernel(
    DeviceGraphView graph, const float* hidden, std::uint32_t hidden_count,
    float* gate_pressure, std::uint32_t gate_count) {
  const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= graph.edge_count) return;
  const EdgeRecord edge = graph.edges[edge_index];
  if (edge.source >= hidden_count || edge.target >= gate_count) return;
  if (edge.source == kInvalidEdge || edge.target == kInvalidEdge) return;
  const std::int32_t support = signed_support(graph, edge_index, edge);
  if (support == 0) return;
  atomicAdd(&gate_pressure[edge.target],
            hidden[edge.source] * static_cast<float>(support) *
                graph.quantum_scale);
}

__device__ inline std::uint64_t hash_graph_state(
    const DeviceGraphView graph) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::uint32_t index = 0u; index < graph.edge_count; ++index) {
    const EdgeRecord edge = graph.edges[index];
    hash = mix_word(hash, edge.source, index * 4u);
    hash = mix_word(hash, edge.target, index * 4u + 1u);
    hash = mix_word(hash, edge.positive_slot, index * 4u + 2u);
    hash = mix_word(hash, edge.negative_slot, index * 4u + 3u);
  }
  const std::uint32_t word_count = graph_route_word_count(graph);
  for (std::uint32_t index = 0u; index < word_count; ++index)
    hash = mix_word(hash, graph.route_words[index],
                    graph.edge_count * 4u + index);
  for (std::uint32_t index = 0u; index < word_count; ++index)
    hash = mix_word(hash, graph.lesion_escrow[index],
                    graph.edge_count * 4u + word_count + index);
  if (graph.development_escrow != nullptr) {
    const B2MutationEscrow escrow = *graph.development_escrow;
    hash = mix_word(hash, escrow.old_edge.source, 0x400u);
    hash = mix_word(hash, escrow.old_edge.target, 0x401u);
    hash = mix_word(hash, escrow.old_edge.positive_slot, 0x402u);
    hash = mix_word(hash, escrow.old_edge.negative_slot, 0x403u);
    hash = mix_word(hash, escrow.live_edge_index, 0x404u);
    hash = mix_word(hash, escrow.positive_word, 0x405u);
    hash = mix_word(hash, escrow.negative_word, 0x406u);
    hash = mix_word(hash, escrow.active, 0x407u);
  }
  return hash;
}

__device__ inline std::uint32_t graph_matter_bits(
    const DeviceGraphView graph) {
  std::uint32_t bits = 0u;
  const std::uint32_t word_count = graph_route_word_count(graph);
  for (std::uint32_t index = 0u; index < word_count; ++index) {
    bits += __popc(graph.route_words[index]);
    bits += __popc(graph.lesion_escrow[index]);
  }
  if (graph.development_escrow != nullptr &&
      graph.development_escrow->active != 0u) {
    bits += __popc(graph.development_escrow->positive_word);
    bits += __popc(graph.development_escrow->negative_word);
  }
  return bits;
}

static __global__ void hash_graph_state_kernel(
    DeviceGraphView graph, std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *output = hash_graph_state(graph);
}

__device__ inline std::uint32_t select_development_edge(
    const DeviceGraphView graph, SiteWord residual) {
  std::uint32_t selected = kInvalidEdge;
  std::uint32_t best_score = 0xffffffffu;
  for (std::uint32_t index = 0u; index < graph.edge_count; ++index) {
    const EdgeRecord edge = graph.edges[index];
    if (edge.source == kInvalidEdge || edge.target == kInvalidEdge) continue;
    if (graph.route_words[edge.positive_slot] == 0u &&
        graph.route_words[edge.negative_slot] == 0u)
      continue;
    const std::uint32_t score = b2_edge_key(edge, index) ^ residual;
    if (selected == kInvalidEdge || score < best_score) {
      selected = index;
      best_score = score;
    }
  }
  return selected;
}

// Forward growth moves the route's signed matter into a device escrow overlay
// while replacing only its endpoints. Reverse consumes that same escrow and
// restores the original edge and matter exactly.
static __global__ void mutate_developmental_route_kernel(
    DeviceGraphView graph, SiteWord residual, std::uint32_t reverse,
    B2MutationReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  B2MutationReceipt local{};
  local.state_before_hash = hash_graph_state(graph);
  local.matter_before_bits = graph_matter_bits(graph);
  if (graph.development_escrow == nullptr) {
    local.state_after_hash = local.state_before_hash;
    local.matter_after_bits = local.matter_before_bits;
    *receipt = local;
    return;
  }
  if (reverse != 0u) {
    B2MutationEscrow escrow = *graph.development_escrow;
    if (escrow.active != 0u && escrow.live_edge_index < graph.edge_count) {
      const EdgeRecord current = graph.edges[escrow.live_edge_index];
      if (current.source != kInvalidEdge && current.target != kInvalidEdge &&
          graph.lesion_escrow[current.positive_slot] == 0u &&
          graph.lesion_escrow[current.negative_slot] == 0u) {
        graph.route_words[escrow.old_edge.positive_slot] =
            escrow.positive_word;
        graph.route_words[escrow.old_edge.negative_slot] =
            escrow.negative_word;
        graph.edges[escrow.live_edge_index] = escrow.old_edge;
        *graph.development_escrow = B2MutationEscrow{};
        local.selected_edge = escrow.live_edge_index;
        local.reversed = 1u;
        local.moved_quanta = __popc(escrow.positive_word) +
                             __popc(escrow.negative_word);
      }
    }
  } else if (residual != 0u && graph.development_escrow->active == 0u &&
             graph.source_count > 1u && graph.target_count > 1u) {
    const std::uint32_t selected = select_development_edge(graph, residual);
    if (selected != kInvalidEdge) {
      const EdgeRecord old_edge = graph.edges[selected];
      const SiteWord positive = graph.route_words[old_edge.positive_slot];
      const SiteWord negative = graph.route_words[old_edge.negative_slot];
      const std::uint32_t seed = b2_edge_key(old_edge, selected) ^ residual;
      B2MutationEscrow next{};
      next.old_edge = old_edge;
      next.live_edge_index = selected;
      next.positive_word = positive;
      next.negative_word = negative;
      next.active = 1u;
      graph.route_words[old_edge.positive_slot] = 0u;
      graph.route_words[old_edge.negative_slot] = 0u;
      EdgeRecord grown = old_edge;
      const std::uint32_t source_delta = 1u +
          (seed % (graph.source_count - 1u));
      const std::uint32_t target_delta = 1u +
          ((seed >> 8u) % (graph.target_count - 1u));
      grown.source = (old_edge.source + source_delta) % graph.source_count;
      grown.target = (old_edge.target + target_delta) % graph.target_count;
      graph.edges[selected] = grown;
      *graph.development_escrow = next;
      local.selected_edge = selected;
      local.new_source = grown.source;
      local.new_target = grown.target;
      local.moved_quanta = __popc(positive) + __popc(negative);
      local.grew = 1u;
    }
  }
  local.state_after_hash = hash_graph_state(graph);
  local.matter_after_bits = graph_matter_bits(graph);
  *receipt = local;
}

static __global__ void hash_graph_matter_kernel(
    const SiteWord* words, std::uint32_t word_count, std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint64_t hash = 1469598103934665603ull;
  for (std::uint32_t index = 0u; index < word_count; ++index)
    hash = mix_word(hash, words[index], index);
  *output = hash;
}

// A lesion is a conserved exchange into device escrow. The same kernel shape
// restores it; no host-side route flag participates in recurrent projection.
static __global__ void lesion_edges_kernel(
    DeviceGraphView graph, const std::uint32_t* edge_ids,
    std::uint32_t lesion_count, GraphReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  GraphReceipt local{};
  const std::uint32_t word_count = graph_route_word_count(graph);
  for (std::uint32_t index = 0u; index < word_count; ++index)
    local.matter_before = mix_word(
        local.matter_before, graph.route_words[index], index);
  for (std::uint32_t index = 0u; index < lesion_count; ++index) {
    const std::uint32_t edge_id = edge_ids[index];
    if (edge_id >= graph.edge_count) continue;
    const EdgeRecord edge = graph.edges[edge_id];
    const std::uint32_t slots[2] = {edge.positive_slot, edge.negative_slot};
    for (std::uint32_t side = 0u; side < 2u; ++side) {
      const std::uint32_t slot = slots[side];
      if (graph.lesion_escrow[slot] != 0u) continue;
      SiteWord matter = graph.route_words[slot];
      if (graph.development_escrow != nullptr &&
          graph.development_escrow->active != 0u &&
          graph.development_escrow->live_edge_index == edge_id) {
        matter = side == 0u ? graph.development_escrow->positive_word
                            : graph.development_escrow->negative_word;
        if (side == 0u)
          graph.development_escrow->positive_word = 0u;
        else
          graph.development_escrow->negative_word = 0u;
      } else {
        graph.route_words[slot] = 0u;
      }
      graph.lesion_escrow[slot] = matter;
      local.moved_quanta += __popc(matter);
    }
  }
  for (std::uint32_t index = 0u; index < word_count; ++index) {
    local.matter_after =
        mix_word(local.matter_after, graph.route_words[index], index);
    local.escrow_after =
        mix_word(local.escrow_after, graph.lesion_escrow[index], index);
  }
  *receipt = local;
}

static __global__ void restore_edges_kernel(
    DeviceGraphView graph, const std::uint32_t* edge_ids,
    std::uint32_t lesion_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  for (std::uint32_t index = 0u; index < lesion_count; ++index) {
    const std::uint32_t edge_id = edge_ids[index];
    if (edge_id >= graph.edge_count) continue;
    const EdgeRecord edge = graph.edges[edge_id];
    const std::uint32_t slots[2] = {edge.positive_slot, edge.negative_slot};
    for (std::uint32_t side = 0u; side < 2u; ++side) {
      const std::uint32_t slot = slots[side];
      if (graph.development_escrow != nullptr &&
          graph.development_escrow->active != 0u &&
          graph.development_escrow->live_edge_index == edge_id) {
        if (side == 0u)
          graph.development_escrow->positive_word |=
              graph.lesion_escrow[slot];
        else
          graph.development_escrow->negative_word |=
              graph.lesion_escrow[slot];
      } else {
        graph.route_words[slot] |= graph.lesion_escrow[slot];
      }
      graph.lesion_escrow[slot] = 0u;
    }
  }
}

}  // namespace bcc32_b1_resident_sparse_recurrent_graph
