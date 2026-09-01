#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace bcc32_cuda_adult_v1 {

inline void cuda_require(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(std::size_t count) { allocate(count); }
  ~DeviceArray() { reset(); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&& other) noexcept { *this = std::move(other); }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = other.pointer_;
      count_ = other.count_;
      capacity_ = other.capacity_;
      other.pointer_ = nullptr;
      other.count_ = 0u;
      other.capacity_ = 0u;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    reset();
    count_ = count;
    capacity_ = count;
    if (count != 0u) {
      cuda_require(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                   "cudaMalloc bcc32 adult v1 state");
    }
  }

  // Grow-only per-contact scratch renewal: keeps the allocation while the
  // requested extent fits, grows zero-filled otherwise. Steady-state
  // contacts stop paying cudaMalloc/cudaFree while first-touch bytes stay
  // deterministic.
  DeviceArray& renew(std::size_t count) {
    if (count > capacity_) {
      reset();
      count_ = count;
      capacity_ = count;
      if (count != 0u) {
        cuda_require(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
            "cudaMalloc bcc32 adult v1 scratch");
        cuda_require(cudaMemset(pointer_, 0, count * sizeof(T)),
                     "zero bcc32 adult v1 scratch");
      }
    } else {
      count_ = count;
    }
    return *this;
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
    pointer_ = nullptr;
    count_ = 0u;
    capacity_ = 0u;
  }

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0u;
  std::size_t capacity_ = 0u;
};

}  // namespace bcc32_cuda_adult_v1
