#ifndef HARDWARE_NATIVE_DIRECT_ADULT_BOOTSTRAP_ONCE_RECURRENCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_BOOTSTRAP_ONCE_RECURRENCE_CUH

// g.bootstrap_once_device_recurrence (#1584).
// One birth bootstrap ignites resident bounded Recipe execution: each device
// cycle executes already-bound Recipe IR over immutable evidence.  The
// tail node alone decides continuation -- it relaunches the instantiated
// graph via cudaStreamGraphTailLaunch while accepted-execution progress
// continues, stops fail-closed on quiescence, and honors the birth-capped
// cycle budget.  Post-birth the host holds no semantic or output authority:
// re-birth and semantic program injection refuse, and only observer receipts
// escape to host.

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kRecurrenceThreads = 128u;
inline constexpr std::uint32_t kRecurrenceBirthSeal = 0x626f6e63ull;

struct RecurrenceControl {
  cudaGraphExec_t exec;
  std::uint64_t device_tail_launches;
  std::uint64_t completed_cycles;
  std::uint64_t chain_violations;
  std::uint64_t last_accepted;  // progress watermark for quiescence
  std::uint32_t max_cycles;     // fixed at birth, never renegotiated
  std::uint32_t quiescent;      // set by the device tail only
  std::uint32_t birth_seal;
};
static_assert(std::is_trivial_v<RecurrenceControl> &&
              std::is_standard_layout_v<RecurrenceControl>);

struct RecurrenceReceipt {
  std::uint64_t cycle_digest;
  std::uint64_t accepted_executions;
  std::uint64_t refused_executions;
};
static_assert(std::is_trivial_v<RecurrenceReceipt> &&
              std::is_standard_layout_v<RecurrenceReceipt>);

struct RecurrenceWorkload {
  const ResidentRecipeIrProgram* programs;
  const ResidentRecipeIrEvidence* evidence;
  ResidentRecipeIrResult* results;
  RecurrenceReceipt* receipt;
  RecurrenceControl* control;
  std::uint32_t count;
};
static_assert(std::is_trivial_v<RecurrenceWorkload> &&
              std::is_standard_layout_v<RecurrenceWorkload>);

struct DirectRecurrenceTopology {
  std::uint32_t stage_nodes;
  std::uint32_t instantiate_count;
  std::uint32_t upload_count;
  std::uint32_t birth_host_launches;
  std::uintptr_t executable_identity;
};
static_assert(std::is_trivial_v<DirectRecurrenceTopology> &&
              std::is_standard_layout_v<DirectRecurrenceTopology>);

// Host launch ledger: steady-state proof is the zero delta of this counter
// across N device-owned cycles.
inline std::uint64_t g_recurrence_host_graph_launches = 0u;

inline cudaError_t direct_recurrence_host_graph_launch(cudaGraphExec_t executable,
                                                       cudaStream_t stream) {
  ++g_recurrence_host_graph_launches;
  return ::cudaGraphLaunch(executable, stream);
}

__device__ inline std::uint64_t direct_recurrence_fold_word(
    std::uint64_t identity, std::uint64_t word) {
  identity ^= word + 0x9e3779b97f4a7c15ull + (identity << 6) + (identity >> 2);
  return identity == 0u ? 1u : identity;
}

template <std::uint32_t Stage>
__global__ void direct_recurrence_cycle_kernel(RecurrenceWorkload* workload) {
  __shared__ std::uint64_t partial[kRecurrenceThreads];
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid =
      workload != nullptr && tid < kRecurrenceThreads &&
      workload->programs != nullptr && workload->evidence != nullptr &&
      workload->results != nullptr &&
      workload->receipt != nullptr && workload->control != nullptr &&
      workload->count != 0u;
  if (!valid) return;
  RecurrenceControl* control = workload->control;
  if (control->birth_seal != kRecurrenceBirthSeal) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->chain_violations), 1ULL);
    return;
  }
  std::uint64_t local = 0u;
  for (std::uint32_t i = tid; i < workload->count; i += kRecurrenceThreads) {
    const bool ok = execute_resident_recipe_ir(workload->programs[i],
                                               workload->evidence[i],
                                               &workload->results[i]);
    if (ok)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &workload->receipt->accepted_executions),
                1ULL);
    else
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &workload->receipt->refused_executions),
                1ULL);
    local = direct_recurrence_fold_word(local,
                                        workload->results[i].execution_identity);
  }
  partial[tid] = local;
  __syncthreads();
  if (tid == 0u) {
    std::uint64_t folded = 0u;
    for (std::uint32_t lane = 0u; lane < kRecurrenceThreads; ++lane)
      folded = direct_recurrence_fold_word(folded, partial[lane]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->cycle_digest),
              static_cast<unsigned long long>(folded));
  }
}

// The device-owned continuation decision: relay while accepted-execution
// progress continues and the birth budget lasts; a stalled dynamics fold is
// declared quiescent and stops the recurrence without host involvement.
template <std::uint32_t NextStage>
__global__ void direct_recurrence_tail_kernel(RecurrenceWorkload* workload) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || workload == nullptr ||
      workload->control == nullptr || workload->receipt == nullptr)
    return;
  RecurrenceControl* control = workload->control;
  ++control->completed_cycles;
  const bool progressed =
      workload->receipt->accepted_executions != control->last_accepted;
  control->last_accepted = workload->receipt->accepted_executions;
  if (!progressed) control->quiescent = 1u;
  if (!progressed || control->completed_cycles >= control->max_cycles) return;
  const cudaError_t status =
      cudaGraphLaunch(control->exec, cudaStreamGraphTailLaunch);
  if (status == cudaSuccess)
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->device_tail_launches),
              1ULL);
  else
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->chain_violations), 1ULL);
}

class BootstrapOnceDeviceRecurrence {
 public:
  BootstrapOnceDeviceRecurrence() = default;
  BootstrapOnceDeviceRecurrence(const BootstrapOnceDeviceRecurrence&) = delete;
  BootstrapOnceDeviceRecurrence& operator=(
      const BootstrapOnceDeviceRecurrence&) = delete;
  ~BootstrapOnceDeviceRecurrence() { destroy(); }

  // Upload-once construction plus the single birth ignition launch.
  cudaError_t birth(RecurrenceWorkload* workload, std::uint32_t max_cycles,
                    cudaStream_t stream = nullptr) {
    if (born_ || workload == nullptr || max_cycles == 0u ||
        workload->programs == nullptr || workload->evidence == nullptr ||
        workload->results == nullptr || workload->receipt == nullptr ||
        workload->count == 0u) {
      ++birth_refusals_;
      return cudaErrorInitializationError;
    }
    cudaError_t status = cudaMallocManaged(&control_, sizeof(*control_));
    if (status != cudaSuccess) return status;
    *control_ = RecurrenceControl{};
    control_->max_cycles = max_cycles;
    workload->control = control_;
    *workload->receipt = RecurrenceReceipt{};
    status = build(workload, stream);
    if (status != cudaSuccess) {
      destroy();
      return status;
    }
    born_ = true;
    control_->birth_seal = kRecurrenceBirthSeal;
    ++topology_.birth_host_launches;
    return direct_recurrence_host_graph_launch(control_->exec, stream);
  }

  // Observer-side drain: blocks until the resident recurrence settles.
  cudaError_t settle(cudaStream_t stream = nullptr) const {
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) return status;
    return cudaGetLastError();
  }

  // Post-birth host semantic authority probe: always refused fail-closed.
  cudaError_t inject_semantic_program(const ResidentRecipeIrProgram&,
                                      std::uint32_t) {
    if (!born_) return cudaErrorInvalidResourceHandle;
    ++injection_refusals_;
    return cudaErrorNotSupported;
  }

  void destroy() noexcept {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
    executable_ = nullptr;
    if (control_ != nullptr) (void)cudaFree(control_);
    control_ = nullptr;
    born_ = false;
  }

  [[nodiscard]] const DirectRecurrenceTopology& topology() const {
    return topology_;
  }
  // Observer access only: the host never regains output authority.
  [[nodiscard]] const RecurrenceControl* control() const { return control_; }
  [[nodiscard]] std::uint32_t birth_refusals() const { return birth_refusals_; }
  [[nodiscard]] std::uint32_t injection_refusals() const {
    return injection_refusals_;
  }

 private:
  cudaError_t build(RecurrenceWorkload* workload, cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    cudaError_t status = cudaGraphCreate(&graph, 0u);
    if (status != cudaSuccess) return status;
    void* arguments[] = {&workload};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(direct_recurrence_cycle_kernel<0u>);
    params.gridDim = dim3{1u, 1u, 1u};
    params.blockDim = dim3{kRecurrenceThreads, 1u, 1u};
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;
    params.extra = nullptr;
    cudaGraphNode_t work_node = nullptr;
    status = cudaGraphAddKernelNode(&work_node, graph, nullptr, 0u, &params);
    if (status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return status;
    }
    params.func = reinterpret_cast<void*>(direct_recurrence_tail_kernel<0u>);
    params.blockDim = dim3{1u, 1u, 1u};
    cudaGraphNode_t tail_node = nullptr;
    status = cudaGraphAddKernelNode(&tail_node, graph, &work_node, 1u, &params);
    if (status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return status;
    }
    size_t node_count = 0u;
    status = cudaGraphGetNodes(graph, nullptr, &node_count);
    if (status == cudaSuccess)
      topology_.stage_nodes = static_cast<std::uint32_t>(node_count);
    if (status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return status;
    }
    status = cudaGraphInstantiateWithFlags(&executable_, graph,
                                           cudaGraphInstantiateFlagDeviceLaunch);
    (void)cudaGraphDestroy(graph);
    if (status != cudaSuccess) return status;
    ++topology_.instantiate_count;
    status = cudaGraphUpload(executable_, stream);
    if (status != cudaSuccess) return status;
    ++topology_.upload_count;
    topology_.executable_identity =
        reinterpret_cast<std::uintptr_t>(executable_);
    control_->exec = executable_;
    return cudaSuccess;
  }

  cudaGraphExec_t executable_ = nullptr;
  RecurrenceControl* control_ = nullptr;
  DirectRecurrenceTopology topology_{};
  std::uint32_t birth_refusals_ = 0u;
  std::uint32_t injection_refusals_ = 0u;
  bool born_ = false;
};

}  // namespace substrate::direct_adult_core

#endif
