#include <stdexcept>

#include "bcc32_prediction_residual_route_toggle.cuh"

namespace substrate::bcc32 {
namespace {

__global__ void prediction_residual_route_toggle_kernel(
    const PredictionResidualNeighborhood* before, PredictionResidualNeighborhood* after,
    PredictionResidualRouteToggleReceipt* receipts, std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const PredictionResidualRouteToggleResult result =
      evaluate_prediction_residual_route_toggle(before[index]);
  after[index] = result.after;
  receipts[index] = result.receipt;
}

}  // namespace

void apply_prediction_residual_route_toggle_batch(
    const PredictionResidualNeighborhood* device_before,
    PredictionResidualNeighborhood* device_after,
    PredictionResidualRouteToggleReceipt* device_receipts, std::uint32_t count,
    cudaStream_t stream) {
  if (count == 0u)
    return;
  if (device_before == nullptr || device_after == nullptr || device_receipts == nullptr) {
    throw std::invalid_argument("prediction residual route toggle null batch");
  }
  constexpr std::uint32_t kThreads = 128u;
  const std::uint32_t blocks = (count + kThreads - 1u) / kThreads;
  prediction_residual_route_toggle_kernel<<<blocks, kThreads, 0, stream>>>(
      device_before, device_after, device_receipts, count);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace substrate::bcc32
