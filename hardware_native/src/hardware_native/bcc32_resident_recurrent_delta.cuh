#pragma once

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hardware_native/bcc32_resident_recurrent_material.cuh"

namespace substrate::bcc32::resident_recurrent_delta {

namespace material = resident_recurrent_material;

inline constexpr std::uint32_t kHiddenSize = 384u;
inline constexpr std::uint32_t kRank = 32u;
inline constexpr std::uint32_t kCohortCount = 2u;
inline constexpr char kMagic[8] = {'B', 'C', 'C', 'M', '7', 'D', '\0', '\0'};
inline constexpr std::uint32_t kVersion = 1u;
inline constexpr std::uint32_t kFixtureTensors = 2u;
inline constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kFNVPrime = 1099511628211ull;

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct DeviceTensorView {
  const std::uint32_t* positive = nullptr;
  const std::uint32_t* negative = nullptr;
  std::uint32_t rows = 0u;
  std::uint32_t cols = 0u;
  float scale = 1.0f;
};

// The immutable inference view remains above.  Training owns this explicitly
// mutable spelling so existing callers cannot accidentally acquire write access.
struct MutableDeviceTensorView {
  std::uint32_t* positive = nullptr;
  std::uint32_t* negative = nullptr;
  std::uint32_t rows = 0u;
  std::uint32_t cols = 0u;
  float scale = 1.0f;
};

struct DeviceView {
  DeviceTensorView projection_b{};
  DeviceTensorView adapter_a{};
};

struct MutableDeviceView {
  DeviceTensorView projection_b{};
  DeviceTensorView adapter_a{};
  class AdapterPublicationEndpoint* publication = nullptr;
};

class ResidentDelta;

class AdapterPublicationEndpoint {
 public:
  AdapterPublicationEndpoint() = default;

  material::PublicationReceipt publish(const std::uint32_t* positive,
                                       const std::uint32_t* negative) const;

  void publish_async(const std::uint32_t* positive,
                     const std::uint32_t* negative,
                     material::DevicePublicationReceipt* receipt) const;

  material::PublicationReceipt read(
      const material::DevicePublicationReceipt* receipt) const;

 private:
  explicit AdapterPublicationEndpoint(material::DeviceView view) : view_(view) {}
  material::DeviceView view_{};
  friend class ResidentDelta;
};

class ResidentTensor {
 public:
  ResidentTensor() = default;
  ResidentTensor(const ResidentTensor&) = delete;
  ResidentTensor& operator=(const ResidentTensor&) = delete;

  ResidentTensor(ResidentTensor&& other) noexcept
      : positive_(std::exchange(other.positive_, nullptr)),
        negative_(std::exchange(other.negative_, nullptr)),
        elements_(std::exchange(other.elements_, 0u)),
        rows_(std::exchange(other.rows_, 0u)),
        cols_(std::exchange(other.cols_, 0u)),
        scale_(std::exchange(other.scale_, 1.0f)) {}

  ResidentTensor& operator=(ResidentTensor&& other) noexcept {
    if (this != &other) {
      release();
      positive_ = std::exchange(other.positive_, nullptr);
      negative_ = std::exchange(other.negative_, nullptr);
      elements_ = std::exchange(other.elements_, 0u);
      rows_ = std::exchange(other.rows_, 0u);
      cols_ = std::exchange(other.cols_, 0u);
      scale_ = std::exchange(other.scale_, 1.0f);
    }
    return *this;
  }

  ~ResidentTensor() { release(); }

  static ResidentTensor from_words(
      std::uint32_t rows, std::uint32_t cols, float scale,
      const std::vector<std::uint32_t>& positive,
      const std::vector<std::uint32_t>& negative) {
    const std::size_t words = static_cast<std::size_t>(rows) * cols *
                              kCohortCount;
    if (rows == 0u || cols == 0u || positive.size() != words ||
        negative.size() != words || !std::isfinite(scale) || scale <= 0.0f)
      throw std::runtime_error("invalid BCCM7D tensor payload");
    ResidentTensor tensor;
    tensor.rows_ = rows;
    tensor.cols_ = cols;
    tensor.elements_ = static_cast<std::uint64_t>(rows) * cols;
    tensor.scale_ = scale;
    const std::size_t bytes = words * sizeof(std::uint32_t);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor.positive_), bytes),
                 "allocate BCCM7D positive tensor");
    try {
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor.negative_), bytes),
                   "allocate BCCM7D negative tensor");
      require_cuda(cudaMemcpy(tensor.positive_, positive.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy BCCM7D positive tensor");
      require_cuda(cudaMemcpy(tensor.negative_, negative.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy BCCM7D negative tensor");
    } catch (...) {
      tensor.release();
      throw;
    }
    return tensor;
  }

  DeviceTensorView view() const {
    return {positive_, negative_, rows_, cols_, scale_};
  }

  std::uint64_t elements() const { return elements_; }

  ResidentTensor snapshot() const {
    ResidentTensor copy;
    if (elements_ == 0u) return copy;
    copy.rows_ = rows_;
    copy.cols_ = cols_;
    copy.elements_ = elements_;
    copy.scale_ = scale_;
    const std::size_t bytes = static_cast<std::size_t>(elements_) *
                              kCohortCount * sizeof(std::uint32_t);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&copy.positive_), bytes),
                 "allocate BCCM7D tensor snapshot positive");
    try {
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&copy.negative_), bytes),
                   "allocate BCCM7D tensor snapshot negative");
      require_cuda(cudaMemcpy(copy.positive_, positive_, bytes,
                              cudaMemcpyDeviceToDevice),
                   "snapshot BCCM7D positive tensor");
      require_cuda(cudaMemcpy(copy.negative_, negative_, bytes,
                              cudaMemcpyDeviceToDevice),
                   "snapshot BCCM7D negative tensor");
    } catch (...) {
      copy.release();
      throw;
    }
    return copy;
  }

  void restore_from(const ResidentTensor& snapshot) {
    if (rows_ != snapshot.rows_ || cols_ != snapshot.cols_ ||
        elements_ != snapshot.elements_ || scale_ != snapshot.scale_)
      throw std::runtime_error("BCCM7D snapshot shape mismatch");
    const std::size_t bytes = static_cast<std::size_t>(elements_) *
                              kCohortCount * sizeof(std::uint32_t);
    require_cuda(cudaMemcpy(positive_, snapshot.positive_, bytes,
                            cudaMemcpyDeviceToDevice),
                 "restore BCCM7D positive tensor");
    require_cuda(cudaMemcpy(negative_, snapshot.negative_, bytes,
                            cudaMemcpyDeviceToDevice),
                 "restore BCCM7D negative tensor");
  }

  void clear() {
    const std::size_t bytes = static_cast<std::size_t>(elements_) *
                              kCohortCount * sizeof(std::uint32_t);
    require_cuda(cudaMemset(positive_, 0, bytes), "clear BCCM7D positive tensor");
    require_cuda(cudaMemset(negative_, 0, bytes), "clear BCCM7D negative tensor");
  }

  void append_hash(std::uint64_t& hash) const {
    const std::array<std::uint32_t, 5> metadata{
        rows_, cols_, static_cast<std::uint32_t>(elements_),
        static_cast<std::uint32_t>(elements_ >> 32u),
        std::bit_cast<std::uint32_t>(scale_)};
    for (const std::uint32_t word : metadata) {
      hash ^= word;
      hash *= kFNVPrime;
    }
    const std::size_t words = static_cast<std::size_t>(elements_) *
                              kCohortCount;
    std::vector<std::uint32_t> positive(words), negative(words);
    const std::size_t bytes = words * sizeof(std::uint32_t);
    require_cuda(cudaMemcpy(positive.data(), positive_, bytes,
                            cudaMemcpyDeviceToHost),
                 "read BCCM7D positive tensor hash");
    require_cuda(cudaMemcpy(negative.data(), negative_, bytes,
                            cudaMemcpyDeviceToHost),
                 "read BCCM7D negative tensor hash");
    for (const std::uint32_t word : positive) {
      hash ^= word;
      hash *= kFNVPrime;
    }
    for (const std::uint32_t word : negative) {
      hash ^= word;
      hash *= kFNVPrime;
    }
  }

 private:
  MutableDeviceTensorView mutable_view() {
    return {positive_, negative_, rows_, cols_, scale_};
  }

  friend class ResidentDelta;

  void release() noexcept {
    if (negative_ != nullptr) (void)cudaFree(negative_);
    if (positive_ != nullptr) (void)cudaFree(positive_);
    negative_ = nullptr;
    positive_ = nullptr;
  }

  std::uint32_t* positive_ = nullptr;
  std::uint32_t* negative_ = nullptr;
  std::uint64_t elements_ = 0u;
  std::uint32_t rows_ = 0u;
  std::uint32_t cols_ = 0u;
  float scale_ = 1.0f;
};

__device__ inline float weight(DeviceTensorView tensor, std::uint32_t row,
                               std::uint32_t column) {
  if (tensor.positive == nullptr || tensor.negative == nullptr ||
      row >= tensor.rows || column >= tensor.cols)
    return 0.0f;
  const std::uint64_t offset =
      (static_cast<std::uint64_t>(row) * tensor.cols + column) *
      kCohortCount;
  int support = 0;
  for (std::uint32_t cohort = 0u; cohort < kCohortCount; ++cohort) {
    support += __popc(tensor.positive[offset + cohort]);
    support -= __popc(tensor.negative[offset + cohort]);
  }
  return static_cast<float>(support) * tensor.scale;
}

__device__ inline bool valid(DeviceView view) {
  return view.projection_b.rows == kRank &&
         view.projection_b.cols == kHiddenSize &&
         view.adapter_a.rows == kHiddenSize && view.adapter_a.cols == kRank &&
         view.projection_b.positive != nullptr &&
         view.projection_b.negative != nullptr &&
         view.adapter_a.positive != nullptr && view.adapter_a.negative != nullptr;
}

// All callers use one block of kHiddenSize threads. Every caller reaches both
// barriers, including the disabled/invalid path, so the helper is safe to
// place directly after each recurrent event.
__device__ inline void apply_device(
    DeviceView view, float* hidden, float* compressed, float* adapted,
    std::uint32_t enabled, std::uint32_t write_back,
    std::uint32_t* apply_count) {
  const std::uint32_t thread = threadIdx.x;
  const bool active = enabled != 0u && valid(view) && hidden != nullptr &&
                      compressed != nullptr && adapted != nullptr;
  if (active && thread < kRank) {
    float value = 0.0f;
    for (std::uint32_t column = 0u; column < kHiddenSize; ++column)
      value += weight(view.projection_b, thread, column) * hidden[column];
    compressed[thread] = tanhf(value);
  }
  __syncthreads();
  if (thread < kHiddenSize && hidden != nullptr && adapted != nullptr) {
    float value = hidden[thread];
    if (active) {
      for (std::uint32_t column = 0u; column < kRank; ++column)
        value += weight(view.adapter_a, thread, column) * compressed[column];
    }
    adapted[thread] = value;
  }
  __syncthreads();
  if (active && write_back != 0u && thread < kHiddenSize)
    hidden[thread] = adapted[thread];
  __syncthreads();
  if (active && thread == 0u && apply_count != nullptr)
    *apply_count += 1u;
  __syncthreads();
}

class ResidentDelta {
 public:
  static ResidentDelta load(const std::string& path) {
    if constexpr (std::endian::native != std::endian::little)
      throw std::runtime_error("BCCM7D requires a little-endian host");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open BCCM7D fixture: " + path);
    std::array<char, 8> magic{};
    read_exact(input, magic.data(), magic.size(), "magic");
    if (magic != std::array<char, 8>{kMagic[0], kMagic[1], kMagic[2], kMagic[3],
                                     kMagic[4], kMagic[5], kMagic[6], kMagic[7]})
      throw std::runtime_error("invalid BCCM7D magic");
    if (read_u32(input, "version") != kVersion ||
        read_u32(input, "hidden size") != kHiddenSize ||
        read_u32(input, "rank") != kRank ||
        read_u32(input, "cohort count") != kCohortCount ||
        read_u32(input, "tensor count") != kFixtureTensors)
      throw std::runtime_error("BCCM7D fixture header mismatch");

    ResidentDelta delta;
    for (std::uint32_t index = 0u; index < kFixtureTensors; ++index) {
      const std::string expected = index == 0u ? "projection_B" : "adapter_A";
      const std::uint32_t name_length = read_u32(input, "tensor name length");
      if (name_length != expected.size())
        throw std::runtime_error("BCCM7D tensor name length mismatch");
      std::string name(name_length, '\0');
      read_exact(input, name.data(), name.size(), "tensor name");
      if (name != expected)
        throw std::runtime_error("BCCM7D tensor order/name mismatch");
      const std::uint32_t rows = read_u32(input, "tensor rows");
      const std::uint32_t cols = read_u32(input, "tensor columns");
      float scale = 0.0f;
      read_exact(input, &scale, sizeof(scale), "tensor scale");
      const std::uint64_t elements = read_u64(input, "tensor elements");
      const std::uint32_t expected_rows = index == 0u ? kRank : kHiddenSize;
      const std::uint32_t expected_cols = index == 0u ? kHiddenSize : kRank;
      if (rows != expected_rows || cols != expected_cols ||
          elements != static_cast<std::uint64_t>(rows) * cols)
        throw std::runtime_error("BCCM7D tensor shape mismatch");
      const std::uint64_t word_count64 = elements * kCohortCount;
      if (word_count64 > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("BCCM7D tensor word count overflow");
      const std::size_t word_count = static_cast<std::size_t>(word_count64);
      std::vector<std::uint32_t> positive(word_count), negative(word_count);
      read_exact(input, positive.data(), word_count * sizeof(std::uint32_t),
                 "positive cohorts");
      read_exact(input, negative.data(), word_count * sizeof(std::uint32_t),
                 "negative cohorts");
      ResidentTensor tensor = ResidentTensor::from_words(
          rows, cols, scale, positive, negative);
      if (index == 0u)
        delta.projection_b_ = std::move(tensor);
      else
        delta.adapter_a_ = std::move(tensor);
    }
    char trailing = '\0';
    if (input.read(&trailing, 1))
      throw std::runtime_error("BCCM7D fixture has trailing bytes");
    delta.material_.initialize_from_device(delta.adapter_a_.view().positive,
                                           delta.adapter_a_.view().negative);
    return delta;
  }

  DeviceView view() const { return {projection_b_.view(), adapter_a_.view()}; }

  MutableDeviceView mutable_view() {
    publication_endpoint_ = AdapterPublicationEndpoint(material_.view(
        adapter_a_.mutable_view().positive, adapter_a_.mutable_view().negative));
    return {projection_b_.view(), adapter_a_.view(), &publication_endpoint_};
  }

  ResidentDelta snapshot() const {
    ResidentDelta copy;
    copy.projection_b_ = projection_b_.snapshot();
    copy.adapter_a_ = adapter_a_.snapshot();
    copy.material_ = material_.snapshot();
    return copy;
  }

  void restore(const ResidentDelta& snapshot) {
    projection_b_.restore_from(snapshot.projection_b_);
    adapter_a_.restore_from(snapshot.adapter_a_);
    material_.restore(snapshot.material_);
  }

  void clear_adapter() { lesion_adapter(); }

  material::PublicationReceipt inverse_publication() {
    return material::inverse(material_.view(adapter_a_.mutable_view().positive,
                                            adapter_a_.mutable_view().negative));
  }

  material::PublicationReceipt lesion_adapter() {
    return material::lesion(material_.view(adapter_a_.mutable_view().positive,
                                           adapter_a_.mutable_view().negative));
  }

  material::PublicationReceipt restore_adapter_lesion() {
    return material::restore_lesion(material_.view(adapter_a_.mutable_view().positive,
                                                   adapter_a_.mutable_view().negative));
  }

  material::Inspection inspect_material() const {
    return material_.inspect(adapter_a_.view().positive, adapter_a_.view().negative);
  }

  std::uint64_t state_hash() const {
    std::uint64_t hash = kFNVOffset;
    projection_b_.append_hash(hash);
    adapter_a_.append_hash(hash);
    material_.append_hash(hash);
    return hash;
  }

  std::uint64_t projection_hash() const {
    std::uint64_t hash = kFNVOffset;
    projection_b_.append_hash(hash);
    return hash;
  }

  std::uint64_t checkpoint_hash() const { return state_hash(); }

 private:
  static void read_exact(std::ifstream& input, void* destination,
                         std::size_t bytes, const char* label) {
    if (bytes != 0u && !input.read(static_cast<char*>(destination),
                                   static_cast<std::streamsize>(bytes)))
      throw std::runtime_error(std::string("truncated BCCM7D ") + label);
  }

  static std::uint32_t read_u32(std::ifstream& input, const char* label) {
    std::uint32_t value = 0u;
    read_exact(input, &value, sizeof(value), label);
    return value;
  }

  static std::uint64_t read_u64(std::ifstream& input, const char* label) {
    std::uint64_t value = 0u;
    read_exact(input, &value, sizeof(value), label);
    return value;
  }

  ResidentTensor projection_b_;
  ResidentTensor adapter_a_;
  material::ResidentMaterial material_;
  AdapterPublicationEndpoint publication_endpoint_;
};

inline material::PublicationReceipt AdapterPublicationEndpoint::publish(
    const std::uint32_t* positive, const std::uint32_t* negative) const {
  return material::publish(view_, positive, negative);
}

inline void AdapterPublicationEndpoint::publish_async(
    const std::uint32_t* positive, const std::uint32_t* negative,
    material::DevicePublicationReceipt* receipt) const {
  material::publish_async(view_, positive, negative, receipt);
}

inline material::PublicationReceipt AdapterPublicationEndpoint::read(
    const material::DevicePublicationReceipt* receipt) const {
  return material::read_publication_receipt(receipt,
                                             "read adapter publication receipt");
}

}  // namespace substrate::bcc32::resident_recurrent_delta
