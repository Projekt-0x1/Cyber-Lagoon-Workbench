#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bcc32_cuda_distributed_event_tissue {

inline void cuda_require(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }

  void allocate(std::size_t count) {
    cudaFree(data_);
    data_ = nullptr;
    count_ = count;
    if (count != 0u)
      cuda_require(cudaMalloc(&data_, count * sizeof(T)), "allocate tissue");
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

__global__ void encode_multiscale_events_kernel(const std::uint8_t* bytes, std::uint32_t byte_count,
                                                std::uint32_t surface, std::uint32_t population,
                                                const std::uint32_t* enabled,
                                                std::uint32_t* activation) {
  const std::uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  if (start >= byte_count)
    return;
  constexpr std::uint32_t scales[] = {1u, 2u, 4u, 8u, 16u};
  for (std::uint32_t scale : scales) {
    if (start + scale > byte_count)
      continue;
    std::uint32_t signature = 2166136261u ^ mix32(surface + scale * 0x9e3779b9u);
    for (std::uint32_t offset = 0u; offset < scale; ++offset) {
      signature ^= bytes[start + offset];
      signature *= 16777619u;
    }
    for (std::uint32_t projection = 0u; projection < 4u; ++projection) {
      const std::uint32_t cell = mix32(signature + projection * 0x85ebca6bu) % population;
      if (enabled[cell] != 0u)
        atomicAdd(activation + cell, 1u);
    }
  }
}

__global__ void rank_activation_kernel(const std::uint32_t* activation,
                                       const std::uint32_t* enabled, std::uint32_t population,
                                       unsigned long long* ranks, std::uint32_t* identities) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= population)
    return;
  identities[cell] = cell;
  ranks[cell] = enabled[cell] == 0u ? 0ull
                                    : (static_cast<unsigned long long>(activation[cell]) << 32u) |
                                          static_cast<unsigned long long>(0xffffffffu - cell);
}

__global__ void learn_recurrent_pattern_kernel(const std::uint32_t* active,
                                               std::uint32_t active_count, std::uint32_t population,
                                               std::uint32_t* weights,
                                               unsigned long long* free_mass) {
  const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= active_count * active_count)
    return;
  const std::uint32_t source = active[edge / active_count];
  const std::uint32_t target = active[edge % active_count];
  if (source >= population || target >= population)
    return;

  unsigned long long available = atomicAdd(free_mass, 0ull);
  while (available != 0ull) {
    const unsigned long long observed = atomicCAS(free_mass, available, available - 1ull);
    if (observed == available) {
      atomicAdd(weights + static_cast<std::size_t>(source) * population + target, 1u);
      return;
    }
    available = observed;
  }
}

__global__ void score_recurrent_completion_kernel(
    const std::uint32_t* sensory, const std::uint32_t* active, std::uint32_t active_count,
    const std::uint32_t* weights, const std::uint32_t* enabled, std::uint32_t population,
    unsigned long long* ranks, std::uint32_t* identities) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= population)
    return;
  identities[cell] = cell;
  if (enabled[cell] == 0u) {
    ranks[cell] = 0ull;
    return;
  }
  unsigned long long recurrent = 0ull;
  for (std::uint32_t index = 0u; index < active_count; ++index) {
    const std::uint32_t source = active[index];
    recurrent += weights[static_cast<std::size_t>(source) * population + cell];
  }
  const unsigned long long score =
      static_cast<unsigned long long>(sensory[cell]) * 8ull + recurrent;
  ranks[cell] = (score << 32u) | static_cast<unsigned long long>(0xffffffffu - cell);
}

__global__ void lesion_population_kernel(std::uint32_t* enabled, std::uint32_t population,
                                         std::uint32_t modulo, std::uint32_t phase,
                                         std::uint32_t* weights, unsigned long long* free_mass) {
  const std::size_t edge = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t edge_count = static_cast<std::size_t>(population) * population;
  if (edge < edge_count) {
    const std::uint32_t source = static_cast<std::uint32_t>(edge / population);
    const std::uint32_t target = static_cast<std::uint32_t>(edge % population);
    if ((source % modulo) == phase || (target % modulo) == phase) {
      const std::uint32_t displaced = atomicExch(weights + edge, 0u);
      if (displaced != 0u)
        atomicAdd(free_mass, displaced);
    }
  }
  const std::uint32_t cell = static_cast<std::uint32_t>(edge);
  if (cell < population && (cell % modulo) == phase)
    enabled[cell] = 0u;
}

class DistributedEventTissue {
 public:
  DistributedEventTissue(std::uint32_t population = 1024u, std::uint32_t active_count = 64u,
                         unsigned long long represented_mass = 1ull << 24u)
      : population_(population),
        active_count_(active_count),
        represented_mass_(represented_mass),
        enabled_(population),
        activation_(population),
        ranks_(population),
        identities_(population),
        active_(active_count),
        weights_(static_cast<std::size_t>(population) * population),
        free_mass_(1u) {
    if (population == 0u || active_count == 0u || active_count > population) {
      throw std::invalid_argument("invalid distributed event tissue dimensions");
    }
    cuda_require(cudaMemset(enabled_.get(), 1, enabled_.bytes()), "enable tissue cells");
    cuda_require(cudaMemset(weights_.get(), 0, weights_.bytes()), "clear tissue weights");
    cuda_require(cudaMemcpy(free_mass_.get(), &represented_mass_, sizeof(represented_mass_),
                            cudaMemcpyHostToDevice),
                 "initialize tissue mass");
  }

  std::vector<std::uint32_t> encode(const std::vector<std::uint8_t>& bytes,
                                    std::uint32_t surface = 0u) {
    encode_to_active(bytes, surface);
    return read_active();
  }

  void learn(const std::vector<std::uint8_t>& bytes, std::uint32_t surface = 0u,
             std::uint32_t repetitions = 1u) {
    encode_to_active(bytes, surface);
    const std::uint32_t edges = active_count_ * active_count_;
    for (std::uint32_t repeat = 0u; repeat < repetitions; ++repeat) {
      learn_recurrent_pattern_kernel<<<(edges + 255u) / 256u, 256u>>>(
          active_.get(), active_count_, population_, weights_.get(), free_mass_.get());
    }
    cuda_require(cudaGetLastError(), "launch distributed pattern learning");
    // 0X1-284: no explicit sync needed here -- this class never uses an
    // explicit CUDA stream, so every kernel launch and every host read
    // (read_active()/represented_mass(), both synchronous D2H cudaMemcpy or
    // a value-returning thrust::reduce) shares the one legacy default
    // stream and is already correctly ordered without a device-wide sync.
  }

  std::vector<std::uint32_t> complete(const std::vector<std::uint8_t>& bytes,
                                      std::uint32_t surface = 0u, std::uint32_t iterations = 4u) {
    encode_to_active(bytes, surface);
    for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
      score_recurrent_completion_kernel<<<(population_ + 255u) / 256u, 256u>>>(
          activation_.get(), active_.get(), active_count_, weights_.get(), enabled_.get(),
          population_, ranks_.get(), identities_.get());
      cuda_require(cudaGetLastError(), "launch distributed pattern completion");
      sort_and_select();
    }
    // 0X1-284: redundant -- read_active() below performs a synchronous D2H
    // cudaMemcpy on the same legacy default stream as the loop's kernels,
    // which already blocks the host until they complete.
    return read_active();
  }

  void lesion_fraction(std::uint32_t modulo, std::uint32_t phase) {
    if (modulo == 0u || phase >= modulo) {
      throw std::invalid_argument("invalid distributed lesion selector");
    }
    const std::size_t edges = static_cast<std::size_t>(population_) * population_;
    lesion_population_kernel<<<static_cast<std::uint32_t>((edges + 255u) / 256u), 256u>>>(
        enabled_.get(), population_, modulo, phase, weights_.get(), free_mass_.get());
    cuda_require(cudaGetLastError(), "launch distributed population lesion");
    // 0X1-284: no explicit sync needed -- see learn()'s note above; this
    // class only ever uses the legacy default stream.
  }

  unsigned long long represented_mass() const {
    unsigned long long free = 0ull;
    cuda_require(cudaMemcpy(&free, free_mass_.get(), sizeof(free), cudaMemcpyDeviceToHost),
                 "read tissue free mass");
    thrust::device_ptr<const std::uint32_t> begin(weights_.get());
    const unsigned long long occupied = thrust::reduce(
        thrust::device, begin, begin + weights_.size(), 0ull, thrust::plus<unsigned long long>());
    return free + occupied;
  }

  std::uint32_t population() const { return population_; }
  std::uint32_t active_count() const { return active_count_; }

 private:
  void encode_to_active(const std::vector<std::uint8_t>& bytes, std::uint32_t surface) {
    if (bytes.empty())
      throw std::invalid_argument("raw event stream must not be empty");
    DeviceBuffer<std::uint8_t> input(bytes.size());
    cuda_require(cudaMemcpy(input.get(), bytes.data(), input.bytes(), cudaMemcpyHostToDevice),
                 "upload raw event stream");
    cuda_require(cudaMemset(activation_.get(), 0, activation_.bytes()), "clear sensory population");
    encode_multiscale_events_kernel<<<(bytes.size() + 255u) / 256u, 256u>>>(
        input.get(), static_cast<std::uint32_t>(bytes.size()), surface, population_, enabled_.get(),
        activation_.get());
    cuda_require(cudaGetLastError(), "launch raw distributed encoding");
    rank_activation_kernel<<<(population_ + 255u) / 256u, 256u>>>(
        activation_.get(), enabled_.get(), population_, ranks_.get(), identities_.get());
    cuda_require(cudaGetLastError(), "rank raw distributed encoding");
    sort_and_select();
    // 0X1-284: no explicit sync needed -- every caller of this helper either
    // launches further kernels on the same legacy default stream (already
    // stream-ordered) or reads back via a synchronous D2H cudaMemcpy that
    // synchronizes on its own.
  }

  void sort_and_select() {
    thrust::device_ptr<unsigned long long> rank_begin(ranks_.get());
    thrust::device_ptr<std::uint32_t> id_begin(identities_.get());
    thrust::sort_by_key(thrust::device, rank_begin, rank_begin + population_, id_begin,
                        thrust::greater<unsigned long long>());
    cuda_require(
        cudaMemcpy(active_.get(), identities_.get(), active_.bytes(), cudaMemcpyDeviceToDevice),
        "select sparse distributed population");
  }

  std::vector<std::uint32_t> read_active() const {
    std::vector<std::uint32_t> result(active_count_);
    cuda_require(cudaMemcpy(result.data(), active_.get(), active_.bytes(), cudaMemcpyDeviceToHost),
                 "read sparse distributed population");
    std::sort(result.begin(), result.end());
    return result;
  }

  std::uint32_t population_;
  std::uint32_t active_count_;
  unsigned long long represented_mass_;
  DeviceBuffer<std::uint32_t> enabled_;
  DeviceBuffer<std::uint32_t> activation_;
  DeviceBuffer<unsigned long long> ranks_;
  DeviceBuffer<std::uint32_t> identities_;
  DeviceBuffer<std::uint32_t> active_;
  DeviceBuffer<std::uint32_t> weights_;
  DeviceBuffer<unsigned long long> free_mass_;
};

inline std::uint32_t overlap(const std::vector<std::uint32_t>& first,
                             const std::vector<std::uint32_t>& second) {
  std::uint32_t count = 0u;
  std::size_t left = 0u;
  std::size_t right = 0u;
  while (left < first.size() && right < second.size()) {
    if (first[left] == second[right]) {
      ++count;
      ++left;
      ++right;
    } else if (first[left] < second[right]) {
      ++left;
    } else {
      ++right;
    }
  }
  return count;
}

}  // namespace bcc32_cuda_distributed_event_tissue
