#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_GRAPH_CHAIN_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_GRAPH_CHAIN_CUH

// g.device_owned_cuda_graphs (#1573).
// Fixed family execution graphs uploaded once at birth:
// G_root -> G_family -> G_commit -> tail G_root.  Birth constructs three
// two-node kernel graphs, instantiates every executable with the device-
// launch flag, uploads each exactly once, and ignites the loop with a single
// host launch.  Steady-state continuation is device-owned: each stage's tail
// node relaunches the next executable with cudaStreamGraphTailLaunch, and
// G_commit closes the cycle back to G_root while cycle budget remains.  Host
// launch traffic after birth is zero; post-birth re-instantiation attempts
// are refused fail-closed.

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kDeviceGraphChainStages = 3u;
inline constexpr std::uint32_t kChainStageRoot = 0u;
inline constexpr std::uint32_t kChainStageFamily = 1u;
inline constexpr std::uint32_t kChainStageCommit = 2u;
inline constexpr std::uint32_t kChainStageNodes = 2u;
inline constexpr std::uint32_t kChainBirthSeal = 0x64677263u;
inline constexpr std::uint32_t kChainThreads = 128u;

struct DirectChainControl {
  cudaGraphExec_t execs[kDeviceGraphChainStages];
  std::uint64_t device_graph_launches;
  std::uint64_t chain_violations;
  std::uint64_t completed_cycles;
  std::uint32_t target_cycles;
  std::uint32_t continuation_enabled;
  std::uint32_t birth_seal;
};
static_assert(std::is_trivial_v<DirectChainControl> &&
              std::is_standard_layout_v<DirectChainControl>);

struct DirectChainCycleReceipt {
  std::uint64_t staged_programs;
  std::uint64_t accepted_executions;
  std::uint64_t refused_executions;
  std::uint64_t commit_epochs;
  std::uint64_t cycle_digest;
};
static_assert(std::is_trivial_v<DirectChainCycleReceipt> &&
              std::is_standard_layout_v<DirectChainCycleReceipt>);

struct DirectChainWorkload {
  const ResidentRecipeIrProgram* programs;
  const ResidentRecipeIrEvidence* evidence;
  ResidentRecipeIrResult* results;
  DirectChainCycleReceipt* receipt;
  DirectChainControl* control;
  std::uint32_t count;
  std::uint32_t cursor;  // cycle slice selector; advanced by G_commit tail
};
static_assert(std::is_trivial_v<DirectChainWorkload> &&
              std::is_standard_layout_v<DirectChainWorkload>);

struct DirectDeviceGraphChainTopology {
  std::uint32_t stage_nodes[kDeviceGraphChainStages];
  std::uint32_t instantiate_count;
  std::uint32_t upload_count;
  std::uint32_t birth_host_launches;
  std::uintptr_t executable_identities[kDeviceGraphChainStages];
};

// Host launch ledger.  Every host cudaGraphLaunch issued for a chain goes
// through the instrumented choke point below; steady-state proof is the
// zero delta of this counter across N device-owned cycles.
inline std::uint64_t g_direct_chain_host_graph_launches = 0u;

inline cudaError_t direct_chain_host_graph_launch(cudaGraphExec_t executable,
                                                  cudaStream_t stream) {
  ++g_direct_chain_host_graph_launches;
  return ::cudaGraphLaunch(executable, stream);
}

__device__ inline std::uint64_t direct_chain_fold_word(
    std::uint64_t identity, std::uint64_t word) {
  identity ^= word + 0x9e3779b97f4a7c15ull + (identity << 6) + (identity >> 2);
  return identity == 0u ? 1u : identity;
}

template <std::uint32_t Stage>
__global__ void direct_chain_root_stage_kernel(DirectChainWorkload* workload) {
  __shared__ std::uint64_t partial[kChainThreads];
  __shared__ std::uint32_t lane_staged[kChainThreads];
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = workload != nullptr && tid < kChainThreads &&
                     workload->programs != nullptr &&
                     workload->evidence != nullptr &&
                     workload->results != nullptr &&
                     workload->receipt != nullptr &&
                     workload->count != 0u;
  if (!valid) return;
  const std::uint32_t base = workload->cursor * workload->count;
  std::uint64_t local = 0u;
  std::uint32_t staged = 0u;
  for (std::uint32_t i = tid; i < workload->count; i += kChainThreads) {
    if (resident_recipe_ir_intact(workload->programs[i]) &&
        workload->evidence[base + i].binding_identity != 0u)
      ++staged;
    local = direct_chain_fold_word(local, workload->programs[i].program_identity);
  }
  partial[tid] = direct_chain_fold_word(local, staged);
  lane_staged[tid] = staged;
  if (tid == 0u) {
    std::uint64_t folded = 0u;
    std::uint64_t staged_total = 0u;
    for (std::uint32_t lane = 0u; lane < kChainThreads; ++lane) {
      folded = direct_chain_fold_word(folded, partial[lane]);
      staged_total += lane_staged[lane];
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->cycle_digest),
              static_cast<unsigned long long>(folded));
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->staged_programs),
              static_cast<unsigned long long>(staged_total));
  }
}

template <std::uint32_t Stage>
__global__ void direct_chain_family_stage_kernel(
    DirectChainWorkload* workload) {
  __shared__ std::uint64_t partial[kChainThreads];
  __shared__ std::uint32_t lane_accepted[kChainThreads];
  __shared__ std::uint32_t lane_refused[kChainThreads];
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = workload != nullptr && tid < kChainThreads &&
                     workload->programs != nullptr &&
                     workload->evidence != nullptr &&
                     workload->results != nullptr &&
                     workload->receipt != nullptr &&
                     workload->count != 0u;
  if (!valid) return;
  const std::uint32_t base = workload->cursor * workload->count;
  std::uint64_t local = 0u;
  std::uint32_t accepted = 0u;
  std::uint32_t refused = 0u;
  for (std::uint32_t i = tid; i < workload->count; i += kChainThreads) {
    const bool ok =
        execute_resident_recipe_ir(workload->programs[i],
                                   workload->evidence[base + i],
                                   &workload->results[base + i]);
    if (ok)
      ++accepted;
    else
      ++refused;
    local = direct_chain_fold_word(
        local, ok ? workload->results[base + i].execution_identity : 0u);
  }
  partial[tid] = direct_chain_fold_word(local, accepted);
  lane_accepted[tid] = accepted;
  lane_refused[tid] = refused;
  if (tid == 0u) {
    std::uint64_t folded = 0u;
    std::uint64_t accepted_total = 0u;
    std::uint64_t refused_total = 0u;
    for (std::uint32_t lane = 0u; lane < kChainThreads; ++lane) {
      folded = direct_chain_fold_word(folded, partial[lane]);
      accepted_total += lane_accepted[lane];
      refused_total += lane_refused[lane];
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->cycle_digest),
              static_cast<unsigned long long>(folded));
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->accepted_executions),
              static_cast<unsigned long long>(accepted_total));
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->refused_executions),
              static_cast<unsigned long long>(refused_total));
  }
}

template <std::uint32_t Stage>
__global__ void direct_chain_commit_stage_kernel(
    DirectChainWorkload* workload) {
  __shared__ std::uint64_t partial[kChainThreads];
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = workload != nullptr && tid < kChainThreads &&
                     workload->receipt != nullptr && workload->results != nullptr &&
                     workload->count != 0u;
  if (!valid) return;
  std::uint64_t local = 0u;
  const std::uint32_t base = workload->cursor * workload->count;
  for (std::uint32_t i = tid; i < workload->count; i += kChainThreads)
    local = direct_chain_fold_word(
        local, workload->results[base + i].execution_identity);
  partial[tid] = local;
  if (tid == 0u) {
    std::uint64_t folded = 0u;
    for (std::uint32_t lane = 0u; lane < kChainThreads; ++lane)
      folded = direct_chain_fold_word(folded, partial[lane]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->commit_epochs), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(&workload->receipt->cycle_digest),
              static_cast<unsigned long long>(folded));
  }
}

// Stage tails carry the chain: G_root tail -> G_family, G_family tail ->
// G_commit, G_commit tail -> G_root while the device-owned cycle budget
// lasts.  A missing or corrupted birth seal refuses the relay fail-closed.
template <std::uint32_t NextStage>
__global__ void direct_chain_stage_tail_kernel(DirectChainWorkload* workload) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || workload == nullptr ||
      workload->control == nullptr)
    return;
  DirectChainControl* control = workload->control;
  if (control->birth_seal != kChainBirthSeal) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->chain_violations), 1ULL);
    return;
  }
  if constexpr (NextStage == kChainStageRoot) {
    ++control->completed_cycles;
    ++workload->cursor;
    if (control->continuation_enabled == 0u ||
        control->completed_cycles >= control->target_cycles)
      return;
  }
  const cudaError_t status =
      cudaGraphLaunch(control->execs[NextStage], cudaStreamGraphTailLaunch);
  if (status == cudaSuccess)
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->device_graph_launches), 1ULL);
  else
    atomicAdd(reinterpret_cast<unsigned long long*>(&control->chain_violations), 1ULL);
}

class DirectDeviceGraphChain {
 public:
  DirectDeviceGraphChain() = default;
  DirectDeviceGraphChain(const DirectDeviceGraphChain&) = delete;
  DirectDeviceGraphChain& operator=(const DirectDeviceGraphChain&) = delete;
  ~DirectDeviceGraphChain() { destroy(); }

  // Upload-once construction plus the single birth ignition launch.  The
  // device-owned cycle budget is fixed here and never renegotiated by host.
  cudaError_t birth(DirectChainWorkload* workload, std::uint32_t target_cycles,
                    cudaStream_t stream = nullptr) {
    if (born_ || workload == nullptr || target_cycles == 0u ||
        workload->programs == nullptr || workload->evidence == nullptr ||
        workload->results == nullptr || workload->receipt == nullptr) {
      ++birth_refusals_;
      return cudaErrorInitializationError;
    }
    cudaError_t status = cudaMallocManaged(&control_, sizeof(*control_));
    if (status != cudaSuccess) return status;
    *control_ = DirectChainControl{};
    control_->target_cycles = target_cycles;
    control_->continuation_enabled = 1u;
    workload->control = control_;

    status = build_stage(kChainStageFamily, workload, stream);
    if (status != cudaSuccess) { destroy(); return status; }
    status = build_stage(kChainStageCommit, workload, stream);
    if (status != cudaSuccess) { destroy(); return status; }
    status = build_stage(kChainStageRoot, workload, stream);
    if (status != cudaSuccess) { destroy(); return status; }

    control_->birth_seal = kChainBirthSeal;
    born_ = true;
    ++topology_.birth_host_launches;
    status = direct_chain_host_graph_launch(control_->execs[kChainStageRoot], stream);
    return status;
  }

  // Observer-side drain: blocks until the resident chain settles.
  cudaError_t settle(cudaStream_t stream = nullptr) const {
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) return status;
    return cudaGetLastError();
  }

  void destroy() noexcept {
    for (std::uint32_t stage = 0u; stage < kDeviceGraphChainStages; ++stage) {
      if (executables_[stage] != nullptr)
        (void)cudaGraphExecDestroy(executables_[stage]);
      executables_[stage] = nullptr;
    }
    if (control_ != nullptr) (void)cudaFree(control_);
    control_ = nullptr;
    born_ = false;
  }

  [[nodiscard]] const DirectDeviceGraphChainTopology& topology() const {
    return topology_;
  }
  [[nodiscard]] DirectChainControl* control() { return control_; }
  [[nodiscard]] std::uint32_t birth_refusals() const { return birth_refusals_; }

  // The deliberate host-launched variant: one host graph launch driving a
  // single G_root -> G_family -> G_commit pass.  Exists only so the
  // instrumentation has a positive control to detect.
  cudaError_t host_drive(cudaStream_t stream = nullptr) {
    if (!born_) return cudaErrorInvalidResourceHandle;
    return direct_chain_host_graph_launch(control_->execs[kChainStageRoot], stream);
  }

 private:
  cudaError_t add_stage_graph(std::uint32_t stage, cudaGraph_t graph,
                              DirectChainWorkload* workload, void* work_entry,
                              void* tail_entry) {
    void* arguments[] = {&workload};
    cudaKernelNodeParams params{};
    params.func = work_entry;
    params.gridDim = dim3{1u, 1u, 1u};
    params.blockDim = dim3{kChainThreads, 1u, 1u};
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;
    params.extra = nullptr;
    cudaGraphNode_t work_node = nullptr;
    cudaError_t status = cudaGraphAddKernelNode(&work_node, graph, nullptr, 0u, &params);
    if (status != cudaSuccess) return status;
    params.func = tail_entry;
    params.blockDim = dim3{1u, 1u, 1u};
    cudaGraphNode_t tail_node = nullptr;
    status = cudaGraphAddKernelNode(&tail_node, graph, &work_node, 1u, &params);
    if (status != cudaSuccess) return status;
    size_t node_count = 0u;
    status = cudaGraphGetNodes(graph, nullptr, &node_count);
    if (status != cudaSuccess) return status;
    topology_.stage_nodes[stage] = static_cast<std::uint32_t>(node_count);
    return cudaSuccess;
  }

  cudaError_t build_stage(std::uint32_t stage, DirectChainWorkload* workload,
                          cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    cudaError_t status = cudaGraphCreate(&graph, 0u);
    if (status != cudaSuccess) return status;
    switch (stage) {
      case kChainStageRoot:
        status = add_stage_graph(
            stage, graph, workload,
            reinterpret_cast<void*>(direct_chain_root_stage_kernel<0u>),
            reinterpret_cast<void*>(
                direct_chain_stage_tail_kernel<kChainStageFamily>));
        break;
      case kChainStageFamily:
        status = add_stage_graph(
            stage, graph, workload,
            reinterpret_cast<void*>(direct_chain_family_stage_kernel<0u>),
            reinterpret_cast<void*>(
                direct_chain_stage_tail_kernel<kChainStageCommit>));
        break;
      default:
        status = add_stage_graph(
            stage, graph, workload,
            reinterpret_cast<void*>(direct_chain_commit_stage_kernel<0u>),
            reinterpret_cast<void*>(
                direct_chain_stage_tail_kernel<kChainStageRoot>));
        break;
    }
    if (status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return status;
    }
    status = cudaGraphInstantiateWithFlags(&executables_[stage], graph,
                                           cudaGraphInstantiateFlagDeviceLaunch);
    if (status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return status;
    }
    ++topology_.instantiate_count;
    status = cudaGraphUpload(executables_[stage], stream);
    if (status == cudaSuccess) ++topology_.upload_count;
    (void)cudaGraphDestroy(graph);
    if (status != cudaSuccess) return status;
    topology_.executable_identities[stage] =
        reinterpret_cast<std::uintptr_t>(executables_[stage]);
    control_->execs[stage] = executables_[stage];
    return cudaSuccess;
  }

  cudaGraphExec_t executables_[kDeviceGraphChainStages]{};
  DirectChainControl* control_ = nullptr;
  DirectDeviceGraphChainTopology topology_{};
  std::uint32_t birth_refusals_ = 0u;
  bool born_ = false;
};

}  // namespace substrate::direct_adult_core

#endif
