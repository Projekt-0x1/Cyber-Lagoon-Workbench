#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PREINSTANTIATED_FAMILY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PREINSTANTIATED_FAMILY_CUH

// g.preinstantiated_family_graphs (#1570).
// One generic resident Recipe IR interpreter node is instantiated at
// bootstrap.  Every later composition is device data; the host may replace
// pointer/count parameters but cannot choose a semantic kernel or graph.

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

struct DirectPreinstantiatedFamilyOutcome {
  std::uint64_t program_identity;
  std::uint64_t logical_recipe_id;
  std::uint64_t execution_identity;
  std::int32_t parameter_delta_q16;
  std::uint32_t accepted;
};

struct DirectPreinstantiatedFamilyMetrics {
  std::uint64_t resident_ir_executions;
  std::uint64_t resident_ir_refusals;
  std::uint64_t host_semantic_dispatches;
};

static_assert(std::is_trivial_v<DirectPreinstantiatedFamilyOutcome> &&
              std::is_standard_layout_v<DirectPreinstantiatedFamilyOutcome>);
static_assert(std::is_trivial_v<DirectPreinstantiatedFamilyMetrics> &&
              std::is_standard_layout_v<DirectPreinstantiatedFamilyMetrics>);

template <std::uint32_t Family = 0u>
__global__ void direct_preinstantiated_recipe_family_kernel(
    const ResidentRecipeIrProgram* programs,
    const ResidentRecipeIrEvidence* evidence,
    ResidentRecipeIrResult* results,
    DirectPreinstantiatedFamilyOutcome* outcomes,
    DirectPreinstantiatedFamilyMetrics* metrics, std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || programs == nullptr || evidence == nullptr ||
      results == nullptr || outcomes == nullptr || metrics == nullptr)
    return;
  const bool accepted =
      execute_resident_recipe_ir(programs[index], evidence[index],
                                 &results[index]);
  outcomes[index] = {programs[index].program_identity,
                     evidence[index].logical_recipe_id,
                     results[index].execution_identity,
                     results[index].parameter_delta_q16,
                     accepted ? 1u : 0u};
  if (accepted)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->resident_ir_executions),
              1ULL);
  else
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->resident_ir_refusals),
              1ULL);
}

class DirectPreinstantiatedRecipeFamily {
 public:
  DirectPreinstantiatedRecipeFamily() = default;
  DirectPreinstantiatedRecipeFamily(
      const DirectPreinstantiatedRecipeFamily&) = delete;
  DirectPreinstantiatedRecipeFamily& operator=(
      const DirectPreinstantiatedRecipeFamily&) = delete;
  ~DirectPreinstantiatedRecipeFamily() { destroy(); }

  cudaError_t bootstrap() {
    if (executable_ != nullptr) return cudaSuccess;
    void* arguments[] = {&programs_, &evidence_, &results_,
                         &outcomes_, &metrics_, &count_};
    cudaKernelNodeParams params = parameters(arguments);
    cudaError_t status = cudaGraphCreate(&graph_, 0u);
    if (status != cudaSuccess) return status;
    status = cudaGraphAddKernelNode(&node_, graph_, nullptr, 0u, &params);
    if (status != cudaSuccess) {
      destroy();
      return status;
    }
    status = cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0u);
    if (status != cudaSuccess) {
      destroy();
      return status;
    }
    ++instantiate_count_;
    return cudaSuccess;
  }

  cudaError_t launch(const ResidentRecipeIrProgram* programs,
                     const ResidentRecipeIrEvidence* evidence,
                     ResidentRecipeIrResult* results,
                     DirectPreinstantiatedFamilyOutcome* outcomes,
                     DirectPreinstantiatedFamilyMetrics* metrics,
                     std::uint32_t count, cudaStream_t stream = nullptr) {
    if (executable_ == nullptr) return cudaErrorInvalidResourceHandle;
    programs_ = programs;
    evidence_ = evidence;
    results_ = results;
    outcomes_ = outcomes;
    metrics_ = metrics;
    count_ = count;
    void* arguments[] = {&programs_, &evidence_, &results_,
                         &outcomes_, &metrics_, &count_};
    cudaKernelNodeParams params = parameters(arguments);
    cudaError_t status =
        cudaGraphExecKernelNodeSetParams(executable_, node_, &params);
    if (status != cudaSuccess) return status;
    ++parameter_update_count_;
    status = cudaGraphLaunch(executable_, stream);
    if (status == cudaSuccess) ++launch_count_;
    return status;
  }

  void destroy() noexcept {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) (void)cudaGraphDestroy(graph_);
    executable_ = nullptr;
    graph_ = nullptr;
    node_ = nullptr;
  }

  [[nodiscard]] std::uint32_t instantiate_count() const {
    return instantiate_count_;
  }
  [[nodiscard]] std::uint32_t parameter_update_count() const {
    return parameter_update_count_;
  }
  [[nodiscard]] std::uint32_t launch_count() const { return launch_count_; }
  [[nodiscard]] std::uintptr_t executable_identity() const {
    return reinterpret_cast<std::uintptr_t>(executable_);
  }

 private:
  cudaKernelNodeParams parameters(void** arguments) {
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(
        direct_preinstantiated_recipe_family_kernel<0u>);
    params.gridDim = dim3{1u, 1u, 1u};
    params.blockDim = dim3{128u, 1u, 1u};
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;
    params.extra = nullptr;
    return params;
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaGraphNode_t node_ = nullptr;
  const ResidentRecipeIrProgram* programs_ = nullptr;
  const ResidentRecipeIrEvidence* evidence_ = nullptr;
  ResidentRecipeIrResult* results_ = nullptr;
  DirectPreinstantiatedFamilyOutcome* outcomes_ = nullptr;
  DirectPreinstantiatedFamilyMetrics* metrics_ = nullptr;
  std::uint32_t count_ = 0u;
  std::uint32_t instantiate_count_ = 0u;
  std::uint32_t parameter_update_count_ = 0u;
  std::uint32_t launch_count_ = 0u;
};

}  // namespace substrate::direct_adult_core

#endif
