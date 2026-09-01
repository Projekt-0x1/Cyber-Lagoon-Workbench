#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdlib>

#include "hardware_native/direct_execution_fabric.cuh"

namespace substrate::direct_adult {

using namespace nvcuda;

__device__ inline void evaluate_dense_tensor_tile_scalar(
    const DirectDenseTile& tile,
    const std::int8_t* in_vector,
    std::int32_t* out_scores) {
  for (std::uint32_t r = 0; r < kDenseTileDimension; ++r) {
    std::int32_t sum = tile.row_biases[r];
    for (std::uint32_t c = 0; c < kDenseTileDimension; ++c) {
      sum += static_cast<std::int32_t>(tile.weights_int8[r * kDenseTileDimension + c]) *
             static_cast<std::int32_t>(in_vector[c]);
    }
    out_scores[r] = sum;
  }
}

// Warp-cooperative WMMA Tensor Core kernel for Ada sm_89 hardware acceleration
__global__ void wmma_dense_tensor_tile_kernel(
    const DirectDenseTile* tiles,
    const std::int8_t* in_vectors,
    std::int32_t* out_scores,
    std::uint32_t tile_count) {
  const std::uint32_t tile_idx = blockIdx.x;
  if (tile_idx >= tile_count)
    return;

  const DirectDenseTile tile = tiles[tile_idx];
  __shared__ alignas(16) signed char smem_a[256];
  __shared__ alignas(16) signed char smem_b[256];
  __shared__ alignas(16) int smem_c[256];

  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < 256u; i += 32u) {
    smem_a[i] = static_cast<signed char>(tile.weights_int8[i]);
    smem_b[i] = (i < 16u) ? static_cast<signed char>(in_vectors[tile_idx * 16u + i]) : 0;
  }
  __syncthreads();

  if (tid < 32u) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;

    wmma::fill_fragment(c_frag, 0);
    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
  }
  __syncthreads();

  if (tid < 16u) {
    out_scores[tile_idx * 16u + tid] = smem_c[tid * 16] + tile.row_biases[tid];
  }
}

// Warp-cooperative FP16/BF16 WMMA kernel
__global__ void wmma_fp16_bf16_dense_tensor_tile_kernel(
    const DirectDenseTile* tiles,
    const std::int8_t* in_vectors,
    std::int32_t* out_scores,
    std::uint32_t tile_count) {
  const std::uint32_t tile_idx = blockIdx.x;
  if (tile_idx >= tile_count)
    return;

  const DirectDenseTile tile = tiles[tile_idx];
  __shared__ alignas(16) half smem_a[256];
  __shared__ alignas(16) half smem_b[256];
  __shared__ alignas(16) float smem_c[256];

  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < 256u; i += 32u) {
    smem_a[i] = __float2half(static_cast<float>(tile.weights_int8[i]));
    smem_b[i] = (i < 16u) ? __float2half(static_cast<float>(in_vectors[tile_idx * 16u + i]))
                          : __float2half(0.0f);
  }
  __syncthreads();

  if (tid < 32u) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
  }
  __syncthreads();

  if (tid < 16u) {
    out_scores[tile_idx * 16u + tid] =
        static_cast<std::int32_t>(smem_c[tid * 16] + static_cast<float>(tile.row_biases[tid]));
  }
}

void launch_wmma_dense_tensor_tile_eval(
    const DirectDenseTile* d_tile,
    const std::int8_t* d_in,
    std::int32_t* d_out,
    std::uint32_t tile_count,
    bool fp16) {
  if (fp16) {
    wmma_fp16_bf16_dense_tensor_tile_kernel<<<tile_count, 32>>>(d_tile, d_in, d_out, tile_count);
  } else {
    wmma_dense_tensor_tile_kernel<<<tile_count, 32>>>(d_tile, d_in, d_out, tile_count);
  }
}

__device__ bool execute_dense_tensor_tile(
    const DirectBrainV01& brain,
    const DirectExecutionFabricDeviceView& fabric,
    std::uint32_t tile_index,
    const ActivityEvent& event,
    ActivityEvent* out_event,
    AdultCounters* counters) {
  if (tile_index >= 1024u)
    return false;

  const DirectDenseTile tile = fabric.dense_tiles[tile_index];
  if ((tile.state_flags & kExecutionFlagLesioned) != 0u)
    return false;

  // Construct input activation vector from triggering event
  std::int8_t in_vector[16]{};
  bool matched_input = false;
  for (std::uint32_t c = 0; c < 16u; ++c) {
    if (tile.input_nodes[c] == event.node) {
      const std::int32_t w = static_cast<std::int32_t>(event.word);
      in_vector[c] = static_cast<std::int8_t>(w < 1 ? 1 : (w > 127 ? 127 : w));
      matched_input = true;
    }
  }
  if (!matched_input)
    return false;

  std::int32_t scores[16]{};
  evaluate_dense_tensor_tile_scalar(tile, in_vector, scores);

  if (tile.numeric_mode == static_cast<std::uint32_t>(DirectDenseNumericMode::fp16_guarded) ||
      tile.numeric_mode == static_cast<std::uint32_t>(DirectDenseNumericMode::bf16_guarded)) {
    std::int32_t max1 = -9999999;
    std::int32_t max2 = -9999999;
    for (std::uint32_t r = 0; r < 16u; ++r) {
      if (scores[r] > max1) {
        max2 = max1;
        max1 = scores[r];
      } else if (scores[r] > max2) {
        max2 = scores[r];
      }
    }
    const std::int32_t margin = max1 - max2;
    const std::int32_t thresh_dist = abs(max1 - tile.threshold_q16);
    if (margin <= tile.guard_band_q16 || thresh_dist <= tile.guard_band_q16) {
      if (counters != nullptr)
        atomicAdd(&counters->guard_fallbacks, 1u);
    }
  }

  // Find single best row score >= threshold
  std::int32_t best_score = tile.threshold_q16;
  std::uint32_t best_row = kInvalidIndex;
  for (std::uint32_t r = 0; r < 16u; ++r) {
    if (scores[r] > best_score) {
      best_score = scores[r];
      best_row = r;
    }
  }

  if (best_row != kInvalidIndex && best_row < 16u) {
    const std::uint32_t target = tile.output_nodes[best_row];
    if (target < brain.node_count) {
      ActivityEvent succ = event;
      succ.node = target;
      succ.cue_node = event.node;
      succ.origin = CausalOrigin::endogenous_prediction;
      succ.word = static_cast<Word>(best_row + 1u);
      *out_event = succ;
      if (counters != nullptr)
        atomicAdd(&counters->dense_activations, 1u);
      return true;
    }
  }

  return false;
}

}  // namespace substrate::direct_adult
