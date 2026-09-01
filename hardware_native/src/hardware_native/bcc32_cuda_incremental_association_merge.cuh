#pragma once

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace bcc32_cuda_incremental_association_merge {

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceStorage {
 public:
  DeviceStorage() = default;
  explicit DeviceStorage(std::size_t count) { allocate(count); }
  ~DeviceStorage() { reset(); }

  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;

  DeviceStorage(DeviceStorage&& other) noexcept { *this = std::move(other); }
  DeviceStorage& operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = other.pointer_;
      count_ = other.count_;
      other.pointer_ = nullptr;
      other.count_ = 0u;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    reset();
    count_ = count;
    if (count != 0u) {
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                   "allocate incremental association workspace");
    }
  }

  void reset() noexcept {
    if (pointer_ != nullptr) cudaFree(pointer_);
    pointer_ = nullptr;
    count_ = 0u;
  }

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  std::size_t size() const { return count_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename Key>
struct Workspace {
  Workspace(std::uint32_t resident_capacity, std::uint32_t delta_capacity,
            std::uint32_t reclamation_headroom = 0u)
      : resident_capacity(resident_capacity), delta_capacity(delta_capacity),
        reclamation_headroom(reclamation_headroom),
        delta_keys(delta_capacity), delta_counts(delta_capacity),
        unique_keys(delta_capacity), unique_counts(delta_capacity),
        match_indices(delta_capacity), novel_flags(delta_capacity),
        novel_keys(delta_capacity), novel_counts(delta_capacity),
        merged_keys(resident_capacity), merged_counts(resident_capacity),
        reclaim_scores(static_cast<std::size_t>(resident_capacity) + delta_capacity),
        reclaim_keep_flags(resident_capacity),
        reclaim_novel_keep_flags(delta_capacity), reclaimed_counts(2u),
        reclaimed_mass(1u), status(1u) {
    if (reclamation_headroom >= resident_capacity && resident_capacity != 0u) {
      throw std::runtime_error("incremental association reclamation headroom is too large");
    }
  }

  std::uint32_t resident_capacity;
  std::uint32_t delta_capacity;
  std::uint32_t reclamation_headroom;
  DeviceStorage<Key> delta_keys;
  DeviceStorage<std::uint32_t> delta_counts;
  DeviceStorage<Key> unique_keys;
  DeviceStorage<std::uint32_t> unique_counts;
  DeviceStorage<std::uint32_t> match_indices;
  DeviceStorage<std::uint32_t> novel_flags;
  DeviceStorage<Key> novel_keys;
  DeviceStorage<std::uint32_t> novel_counts;
  DeviceStorage<Key> merged_keys;
  DeviceStorage<std::uint32_t> merged_counts;
  DeviceStorage<unsigned long long> reclaim_scores;
  DeviceStorage<std::uint32_t> reclaim_keep_flags;
  DeviceStorage<std::uint32_t> reclaim_novel_keep_flags;
  DeviceStorage<std::uint32_t> reclaimed_counts;
  DeviceStorage<unsigned long long> reclaimed_mass;
  DeviceStorage<std::uint32_t> status;
};

struct MergeReport {
  std::uint32_t input_delta_count = 0u;
  std::uint32_t unique_delta_count = 0u;
  std::uint32_t updated_resident_count = 0u;
  std::uint32_t inserted_resident_count = 0u;
  std::uint32_t reclaimed_resident_count = 0u;
  std::uint32_t reclaimed_delta_count = 0u;
  std::uint32_t output_resident_count = 0u;
  bool capacity_consolidated = false;
  unsigned long long resident_mass_before = 0u;
  unsigned long long delta_mass = 0u;
  unsigned long long reclaimed_mass = 0u;
  unsigned long long resident_mass_after = 0u;
};

struct CountToMass {
  __host__ __device__ unsigned long long operator()(std::uint32_t value) const {
    return value;
  }
};

struct IsNonzero {
  __host__ __device__ bool operator()(std::uint32_t value) const { return value != 0u; }
};

template <typename Key>
__global__ void classify_delta_kernel(
    const Key* resident_keys, const std::uint32_t* resident_counts,
    std::uint32_t resident_count, const Key* delta_keys,
    const std::uint32_t* delta_counts, std::uint32_t delta_count,
    std::uint32_t* match_indices, std::uint32_t* novel_flags,
    std::uint32_t* status) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= delta_count) return;
  const Key wanted = delta_keys[index];
  std::uint32_t lo = 0u;
  std::uint32_t hi = resident_count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (resident_keys[mid] < wanted) lo = mid + 1u;
    else hi = mid;
  }
  if (lo < resident_count && resident_keys[lo] == wanted) {
    match_indices[index] = lo;
    novel_flags[index] = 0u;
    if (resident_counts[lo] > 0xffffffffu - delta_counts[index]) {
      atomicExch(status, 1u);
    }
  } else {
    match_indices[index] = 0xffffffffu;
    novel_flags[index] = 1u;
  }
}

__global__ void apply_existing_delta_kernel(
    std::uint32_t* resident_counts, const std::uint32_t* delta_counts,
    const std::uint32_t* match_indices, std::uint32_t delta_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= delta_count) return;
  const std::uint32_t resident_index = match_indices[index];
  if (resident_index != 0xffffffffu) {
    resident_counts[resident_index] += delta_counts[index];
  }
}

__global__ void build_reclaim_scores_kernel(
    const std::uint32_t* resident_counts, std::uint32_t resident_count,
    unsigned long long* scores) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= resident_count) return;
  scores[index] =
      (static_cast<unsigned long long>(resident_counts[index]) << 32u) | index;
}

__global__ void build_novel_reclaim_scores_kernel(
    const std::uint32_t* novel_counts, std::uint32_t novel_count,
    unsigned long long* scores) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= novel_count) return;
  const std::uint32_t encoded = index | 0x80000000u;
  scores[index] =
      (static_cast<unsigned long long>(novel_counts[index]) << 32u) | encoded;
}

__global__ void mark_reclaimed_kernel(
    const unsigned long long* sorted_scores, std::uint32_t reclaim_count,
    const std::uint32_t* resident_counts, const std::uint32_t* novel_counts,
    std::uint32_t* resident_keep_flags, std::uint32_t* novel_keep_flags,
    std::uint32_t* reclaimed_counts, unsigned long long* reclaimed_mass) {
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= reclaim_count) return;
  const std::uint32_t encoded = static_cast<std::uint32_t>(sorted_scores[rank]);
  const bool novel = (encoded & 0x80000000u) != 0u;
  const std::uint32_t index = encoded & 0x7fffffffu;
  if (novel) {
    novel_keep_flags[index] = 0u;
    atomicAdd(reclaimed_counts + 1u, 1u);
    atomicAdd(reclaimed_mass, static_cast<unsigned long long>(novel_counts[index]));
  } else {
    resident_keep_flags[index] = 0u;
    atomicAdd(reclaimed_counts, 1u);
    atomicAdd(reclaimed_mass, static_cast<unsigned long long>(resident_counts[index]));
  }
}

__global__ void return_reclaimed_mass_kernel(
    const unsigned long long* reclaimed_mass,
    std::uint32_t* resident_mass_reserve,
    std::uint32_t* resident_occupied_mass, std::uint32_t* status) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const unsigned long long mass = reclaimed_mass[0];
  if (mass > 0xffffffffull || resident_occupied_mass[0] < mass ||
      resident_mass_reserve[0] > 0xffffffffull - mass) {
    status[0] = 1u;
    return;
  }
  resident_occupied_mass[0] -= static_cast<std::uint32_t>(mass);
  resident_mass_reserve[0] += static_cast<std::uint32_t>(mass);
}

inline std::uint32_t blocks_for(std::uint32_t count) {
  constexpr std::uint32_t block = 256u;
  return (count + block - 1u) / block;
}

template <typename Key>
MergeReport merge_sorted_delta(
    Key* resident_keys, std::uint32_t* resident_counts,
    std::uint32_t resident_count, std::uint32_t resident_capacity,
    const Key* appended_delta_keys, const std::uint32_t* appended_delta_counts,
    std::uint32_t delta_count, Workspace<Key>& workspace,
    std::uint32_t* resident_mass_reserve = nullptr,
    std::uint32_t* resident_occupied_mass = nullptr) {
  if (resident_count > resident_capacity || resident_capacity > workspace.resident_capacity ||
      delta_count > workspace.delta_capacity) {
    throw std::runtime_error("incremental association merge capacity exceeded");
  }
  if ((resident_mass_reserve == nullptr) != (resident_occupied_mass == nullptr)) {
    throw std::runtime_error("incremental association reclamation ledger is incomplete");
  }

  MergeReport report{};
  report.input_delta_count = delta_count;
  report.output_resident_count = resident_count;
  report.resident_mass_before = thrust::transform_reduce(
      thrust::device, resident_counts, resident_counts + resident_count,
      CountToMass{}, 0ull, thrust::plus<unsigned long long>());
  if (delta_count == 0u) {
    report.resident_mass_after = report.resident_mass_before;
    return report;
  }

  require_cuda(cudaMemcpy(workspace.delta_keys.get(), appended_delta_keys,
                          static_cast<std::size_t>(delta_count) * sizeof(Key),
                          cudaMemcpyDeviceToDevice),
               "copy incremental association delta keys");
  require_cuda(cudaMemcpy(workspace.delta_counts.get(), appended_delta_counts,
                          static_cast<std::size_t>(delta_count) * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice),
               "copy incremental association delta counts");
  report.delta_mass = thrust::transform_reduce(
      thrust::device, workspace.delta_counts.get(),
      workspace.delta_counts.get() + delta_count, CountToMass{}, 0ull,
      thrust::plus<unsigned long long>());

  thrust::sort_by_key(thrust::device, workspace.delta_keys.get(),
                      workspace.delta_keys.get() + delta_count,
                      workspace.delta_counts.get());
  auto reduced_end = thrust::reduce_by_key(
      thrust::device, workspace.delta_keys.get(),
      workspace.delta_keys.get() + delta_count, workspace.delta_counts.get(),
      workspace.unique_keys.get(), workspace.unique_counts.get());
  const std::uint32_t unique_count =
      static_cast<std::uint32_t>(reduced_end.first - workspace.unique_keys.get());
  report.unique_delta_count = unique_count;

  require_cuda(cudaMemset(workspace.status.get(), 0, sizeof(std::uint32_t)),
               "clear incremental association status");
  constexpr std::uint32_t block = 256u;
  classify_delta_kernel<<<blocks_for(unique_count), block>>>(
      resident_keys, resident_counts, resident_count, workspace.unique_keys.get(),
      workspace.unique_counts.get(), unique_count, workspace.match_indices.get(),
      workspace.novel_flags.get(), workspace.status.get());
  require_cuda(cudaGetLastError(), "classify incremental association delta");
  std::uint32_t status = 0u;
  require_cuda(cudaMemcpy(&status, workspace.status.get(), sizeof(status),
                          cudaMemcpyDeviceToHost),
               "read incremental association status");
  if (status != 0u) throw std::runtime_error("incremental association count overflow");

  std::uint32_t novel_count = thrust::reduce(
      thrust::device, workspace.novel_flags.get(),
      workspace.novel_flags.get() + unique_count, 0u, thrust::plus<std::uint32_t>());
  report.updated_resident_count = unique_count - novel_count;
  if (novel_count != 0u) {
    thrust::copy_if(thrust::device, workspace.unique_keys.get(),
                    workspace.unique_keys.get() + unique_count,
                    workspace.novel_flags.get(), workspace.novel_keys.get(), IsNonzero{});
    thrust::copy_if(thrust::device, workspace.unique_counts.get(),
                    workspace.unique_counts.get() + unique_count,
                    workspace.novel_flags.get(), workspace.novel_counts.get(), IsNonzero{});
  }

  apply_existing_delta_kernel<<<blocks_for(unique_count), block>>>(
      resident_counts, workspace.unique_counts.get(), workspace.match_indices.get(),
      unique_count);
  require_cuda(cudaGetLastError(), "apply existing incremental association counts");

  const std::uint32_t target_resident_count =
      resident_capacity - workspace.reclamation_headroom;
  if (novel_count > target_resident_count) {
    throw std::runtime_error(
        "incremental delta exceeds the resident evidence consolidation aperture");
  }
  const std::uint64_t projected_count =
      static_cast<std::uint64_t>(resident_count) + novel_count;
  const Key* merge_novel_keys = workspace.novel_keys.get();
  const std::uint32_t* merge_novel_counts = workspace.novel_counts.get();
  if (projected_count > target_resident_count) {
    const std::uint32_t reclaim_count = static_cast<std::uint32_t>(
        projected_count - target_resident_count);
    build_reclaim_scores_kernel<<<blocks_for(resident_count), block>>>(
        resident_counts, resident_count, workspace.reclaim_scores.get());
    require_cuda(cudaGetLastError(), "build resident evidence reclamation scores");
    if (novel_count != 0u) {
      build_novel_reclaim_scores_kernel<<<blocks_for(novel_count), block>>>(
          workspace.novel_counts.get(), novel_count,
          workspace.reclaim_scores.get() + resident_count);
      require_cuda(cudaGetLastError(), "build incoming evidence reclamation scores");
    }
    thrust::sort(thrust::device, workspace.reclaim_scores.get(),
                 workspace.reclaim_scores.get() + projected_count);
    require_cuda(cudaMemset(workspace.reclaim_keep_flags.get(), 1,
                            static_cast<std::size_t>(resident_count) *
                                sizeof(std::uint32_t)),
                 "initialize resident evidence retention flags");
    require_cuda(cudaMemset(workspace.reclaim_novel_keep_flags.get(), 1,
                            static_cast<std::size_t>(novel_count) *
                                sizeof(std::uint32_t)),
                 "initialize incoming evidence retention flags");
    require_cuda(cudaMemset(workspace.reclaimed_counts.get(), 0,
                            2u * sizeof(std::uint32_t)),
                 "clear resident reclaimed entry counts");
    require_cuda(cudaMemset(workspace.reclaimed_mass.get(), 0,
                            sizeof(unsigned long long)),
                 "clear resident reclaimed mass");
    mark_reclaimed_kernel<<<blocks_for(reclaim_count), block>>>(
        workspace.reclaim_scores.get(), reclaim_count, resident_counts,
        workspace.novel_counts.get(), workspace.reclaim_keep_flags.get(),
        workspace.reclaim_novel_keep_flags.get(),
        workspace.reclaimed_counts.get(), workspace.reclaimed_mass.get());
    require_cuda(cudaGetLastError(), "mark weakest resident evidence for reclamation");
    thrust::copy_if(thrust::device, resident_keys, resident_keys + resident_count,
                    workspace.reclaim_keep_flags.get(), workspace.merged_keys.get(),
                    IsNonzero{});
    thrust::copy_if(thrust::device, resident_counts,
                    resident_counts + resident_count,
                    workspace.reclaim_keep_flags.get(), workspace.merged_counts.get(),
                    IsNonzero{});
    std::uint32_t reclaimed_counts[2] = {};
    require_cuda(cudaMemcpy(reclaimed_counts, workspace.reclaimed_counts.get(),
                            sizeof(reclaimed_counts), cudaMemcpyDeviceToHost),
                 "read resident reclaimed entry counts");
    resident_count -= reclaimed_counts[0];
    require_cuda(cudaMemcpy(resident_keys, workspace.merged_keys.get(),
                            static_cast<std::size_t>(resident_count) * sizeof(Key),
                            cudaMemcpyDeviceToDevice),
                 "publish consolidated resident association keys");
    require_cuda(cudaMemcpy(resident_counts, workspace.merged_counts.get(),
                            static_cast<std::size_t>(resident_count) *
                                sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "publish consolidated resident association counts");
    thrust::copy_if(thrust::device, workspace.novel_keys.get(),
                    workspace.novel_keys.get() + novel_count,
                    workspace.reclaim_novel_keep_flags.get(),
                    workspace.delta_keys.get(), IsNonzero{});
    thrust::copy_if(thrust::device, workspace.novel_counts.get(),
                    workspace.novel_counts.get() + novel_count,
                    workspace.reclaim_novel_keep_flags.get(),
                    workspace.delta_counts.get(), IsNonzero{});
    novel_count -= reclaimed_counts[1];
    merge_novel_keys = workspace.delta_keys.get();
    merge_novel_counts = workspace.delta_counts.get();
    require_cuda(cudaMemcpy(&report.reclaimed_mass, workspace.reclaimed_mass.get(),
                            sizeof(report.reclaimed_mass), cudaMemcpyDeviceToHost),
                 "read resident reclaimed evidence mass");
    if (resident_mass_reserve != nullptr) {
      require_cuda(cudaMemset(workspace.status.get(), 0, sizeof(std::uint32_t)),
                   "clear resident reclamation ledger status");
      const unsigned long long* reclaimed_mass = workspace.reclaimed_mass.get();
      std::uint32_t* mass_reserve = resident_mass_reserve;
      std::uint32_t* occupied_mass = resident_occupied_mass;
      std::uint32_t* ledger_status = workspace.status.get();
      void* kernel_arguments[] = {
          &reclaimed_mass, &mass_reserve, &occupied_mass, &ledger_status};
      require_cuda(
          cudaLaunchKernel(
              reinterpret_cast<const void*>(return_reclaimed_mass_kernel),
              dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, kernel_arguments, 0u,
              nullptr),
          "launch return reclaimed evidence to resident mass sink");
      require_cuda(cudaGetLastError(), "return reclaimed evidence to resident mass sink");
      std::uint32_t reclamation_status = 0u;
      require_cuda(cudaMemcpy(&reclamation_status, workspace.status.get(),
                              sizeof(reclamation_status), cudaMemcpyDeviceToHost),
                   "read resident reclamation ledger status");
      if (reclamation_status != 0u) {
        throw std::runtime_error("incremental association reclamation ledger overflow");
      }
    }
    report.reclaimed_resident_count = reclaimed_counts[0];
    report.reclaimed_delta_count = reclaimed_counts[1];
    report.capacity_consolidated = true;
  }

  report.inserted_resident_count = novel_count;
  if (novel_count != 0u) {
    thrust::merge_by_key(
        thrust::device, resident_keys, resident_keys + resident_count,
        merge_novel_keys, merge_novel_keys + novel_count,
        resident_counts, merge_novel_counts, workspace.merged_keys.get(),
        workspace.merged_counts.get());
    require_cuda(cudaMemcpy(resident_keys, workspace.merged_keys.get(),
                            static_cast<std::size_t>(resident_count + novel_count) * sizeof(Key),
                            cudaMemcpyDeviceToDevice),
                 "publish incrementally merged association keys");
    require_cuda(cudaMemcpy(
                     resident_counts, workspace.merged_counts.get(),
                     static_cast<std::size_t>(resident_count + novel_count) *
                         sizeof(std::uint32_t),
                     cudaMemcpyDeviceToDevice),
                 "publish incrementally merged association counts");
  }

  report.output_resident_count = resident_count + novel_count;
  report.resident_mass_after = thrust::transform_reduce(
      thrust::device, resident_counts,
      resident_counts + report.output_resident_count, CountToMass{}, 0ull,
      thrust::plus<unsigned long long>());
  if (report.resident_mass_after + report.reclaimed_mass !=
      report.resident_mass_before + report.delta_mass) {
    throw std::runtime_error("incremental association mass mismatch");
  }
  return report;
}

}  // namespace bcc32_cuda_incremental_association_merge
