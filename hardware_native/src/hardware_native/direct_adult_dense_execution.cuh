#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DENSE_EXECUTION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DENSE_EXECUTION_CUH

#include <cuda_fp16.h>
#include <mma.h>

#include "hardware_native/direct_adult_device_ops.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kDirectDenseTileElements = 16u * 16u;

#if defined(HARDWARE_NATIVE_DEFINE_DENSE_SHATTER_RUNTIME)
static __device__ std::uint32_t dense_shatter_contributor_mask_impl(
    const DirectDenseBlock& block,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, const std::uint32_t* sparse_links,
    std::uint32_t sparse_count) {
  std::uint32_t mask = 0u;
  for (std::uint32_t i = 0u; i < sparse_count; ++i) {
    const DirectActionParticipationLink& link =
        action_links[participant_offset + sparse_links[i]];
    if (link.source_node >= block.node_begin &&
        link.source_node < block.node_begin + block.node_count)
      mask |= 1u << i;
  }
  return mask;
}

__device__ std::uint32_t record_dense_shatter_transaction(
    DirectDenseBlock* dense_blocks, std::uint32_t dense_block_count,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, const std::uint32_t* sparse_links,
    std::uint32_t sparse_count,
    direct_network::DirectExactHistoryRecord* records,
    std::uint64_t action_ticket, std::uint64_t upstream_ticket,
    std::uint32_t commit_tick, std::uint32_t emission_tick) {
  if (dense_blocks == nullptr || action_links == nullptr ||
      sparse_links == nullptr)
    return 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t b = 0u; b < dense_block_count; ++b) {
    const DirectDenseBlock& block = dense_blocks[b];
    if ((block.flags & direct_network::kDenseBlockFlagTensorEligible) == 0u)
      continue;
    const std::uint32_t contributors = dense_shatter_contributor_mask_impl(
        block, action_links, participant_offset, sparse_links, sparse_count);
    if (contributors == 0u) continue;
    if (records != nullptr) {
      direct_network::DirectExactHistoryRecord record{};
      record.identity = action_ticket;
      record.parent_identity = upstream_ticket;
      record.resident_tick = commit_tick;
      record.event_tick = emission_tick;
      record.kind = direct_network::DirectExactHistoryKind::dense_shatter;
      record.source = b;
      record.subject = block.node_begin;
      record.value = block.node_count;
      record.context = contributors;
      record.flags = block.flags;
      record.incarnation_before = block.flags;
      record.incarnation_after =
          block.flags & ~direct_network::kDenseBlockFlagTensorEligible;
      record.resource_delta =
          static_cast<std::int64_t>(record.incarnation_after) - block.flags;
      records[count] = record;
    }
    ++count;
  }
  return count;
}

__device__ void apply_dense_shatter_transaction(
    DirectDenseBlock* dense_blocks, std::uint32_t dense_block_count,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, const std::uint32_t* sparse_links,
    std::uint32_t sparse_count, AdultCoreMetrics* metrics) {
  if (dense_blocks == nullptr || action_links == nullptr ||
      sparse_links == nullptr)
    return;
  for (std::uint32_t b = 0u; b < dense_block_count; ++b) {
    DirectDenseBlock& block = dense_blocks[b];
    if ((block.flags & direct_network::kDenseBlockFlagTensorEligible) == 0u ||
        dense_shatter_contributor_mask_impl(
            block, action_links, participant_offset, sparse_links,
            sparse_count) == 0u)
      continue;
    block.flags &= ~direct_network::kDenseBlockFlagTensorEligible;
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->dense_shatters),
                1ULL);
  }
}
#endif

// One warp executes one dense block. Both canonical executors call this exact
// primitive so a representation choice cannot change the adult's future.
__device__ inline void execute_dense_tensor_block(
    const DirectNode* nodes, const DirectDenseBlock* dense_blocks,
    std::uint32_t block_index, const std::uint16_t* dense_weights_fp16,
    std::int32_t* node_incoming_excitation,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity, std::uint32_t current_tick,
    AdultCoreMetrics* metrics, std::uint32_t lane_id, half* shared_weights,
    half* shared_activation, float* shared_output) {
  const DirectDenseBlock dense = dense_blocks[block_index];
  if ((dense.flags & direct_network::kDenseBlockFlagTensorEligible) == 0u)
    return;

  const std::uint32_t node_count = dense.node_count;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  if (node_count >= 16u && (node_count % 16u) == 0u) {
    const std::uint32_t tile_count = node_count / 16u;
    for (std::uint32_t row_tile = 0u; row_tile < tile_count; ++row_tile) {
      float accumulated[16] = {0.0f};
      for (std::uint32_t column_tile = 0u; column_tile < tile_count;
           ++column_tile) {
        for (std::uint32_t element = 0u; element < 8u; ++element) {
          const std::uint32_t flat = lane_id * 8u + element;
          const std::uint32_t row = flat / 16u;
          const std::uint32_t column = flat % 16u;
          const std::uint32_t weight =
              dense.weight_offset + (row_tile * 16u + row) * node_count +
              column_tile * 16u + column;
          shared_weights[flat] =
              *reinterpret_cast<const half*>(&dense_weights_fp16[weight]);
        }
        if (lane_id < 16u) {
          const std::uint32_t source =
              dense.node_begin + column_tile * 16u + lane_id;
          const half activation = __float2half(
              static_cast<float>(nodes[source].activation_q16) / 65536.0f);
          for (std::uint32_t column = 0u; column < 16u; ++column)
            shared_activation[lane_id * 16u + column] = activation;
        }
        __syncwarp();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                               nvcuda::wmma::row_major>
            weights;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                               nvcuda::wmma::row_major>
            activation;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            output;
        nvcuda::wmma::load_matrix_sync(weights, shared_weights, 16);
        nvcuda::wmma::load_matrix_sync(activation, shared_activation, 16);
        nvcuda::wmma::fill_fragment(output, 0.0f);
        nvcuda::wmma::mma_sync(output, weights, activation, output);
        nvcuda::wmma::store_matrix_sync(shared_output, output, 16,
                                        nvcuda::wmma::mem_row_major);
        __syncwarp();
        if (lane_id < 16u)
          accumulated[lane_id] += shared_output[lane_id * 16u];
        __syncwarp();
        if (lane_id == 0u && metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->dense_wmma_tiles_executed),
                    1ULL);
      }
      if (lane_id < 16u) {
        const std::int32_t excitation =
            static_cast<std::int32_t>(accumulated[lane_id] * 65536.0f);
        if (excitation != 0) {
          const std::uint32_t target =
              dense.node_begin + row_tile * 16u + lane_id;
          atomicAdd(node_incoming_excitation + target, excitation);
          if (node_next_ancestry_incomplete != nullptr)
            atomicExch(node_next_ancestry_incomplete + target, 1u);
          stage_incomplete_contribution(
              dense.node_begin, target, kInvalidIndex, block_index,
              current_tick, participation_staging, participation_staging_count,
              participation_staging_capacity, metrics);
        }
      }
    }
    return;
  }
#endif

  for (std::uint32_t row = lane_id; row < node_count; row += 32u) {
    float accumulated = 0.0f;
    const std::uint32_t weight_base =
        dense.weight_offset + row * node_count;
    for (std::uint32_t column = 0u; column < node_count; ++column) {
      const float activation =
          static_cast<float>(nodes[dense.node_begin + column].activation_q16) /
          65536.0f;
      if (activation > 0.0001f) {
        const half weight = *reinterpret_cast<const half*>(
            &dense_weights_fp16[weight_base + column]);
        accumulated += __half2float(weight) * activation;
      }
    }
    const std::int32_t excitation =
        static_cast<std::int32_t>(accumulated * 65536.0f);
    if (excitation != 0) {
      const std::uint32_t target = dense.node_begin + row;
      atomicAdd(node_incoming_excitation + target, excitation);
      if (node_next_ancestry_incomplete != nullptr)
        atomicExch(node_next_ancestry_incomplete + target, 1u);
      stage_incomplete_contribution(
          dense.node_begin, target, kInvalidIndex, block_index, current_tick,
          participation_staging, participation_staging_count,
          participation_staging_capacity, metrics);
    }
  }
  if (lane_id == 0u && metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->dense_scalar_tiles_executed),
              1ULL);
}

}  // namespace substrate::direct_adult_core

#endif
