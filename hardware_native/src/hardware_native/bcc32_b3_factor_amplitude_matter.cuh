#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

namespace bcc32_b3_factor_amplitude_matter {

using substrate::bcc32::SiteWord;

constexpr std::uint32_t kBatch = 32u;
constexpr std::uint32_t kHidden = 384u;
constexpr std::uint32_t kInput = 96u;
constexpr std::uint32_t kFactorFamilies = 4u;
constexpr std::uint32_t kFamilyCount = 6u;
constexpr std::uint32_t kHorizon = 4u;
constexpr std::uint32_t kFixtureContacts = 6u;
constexpr std::uint32_t kFactorWords = kBatch * kHidden;
constexpr std::uint32_t kInputWords = kBatch * kInput;
constexpr std::uint32_t kPreviousWords = kBatch * kHidden;
constexpr std::uint32_t kPopulationWords =
    kFactorFamilies * kFactorWords + kInputWords + kPreviousWords;
constexpr std::uint32_t kRingWords = kHorizon * kPopulationWords;
constexpr std::uint32_t kStateSegments = kFixtureContacts + 1u + 2u * kHorizon + 1u;

static_assert(sizeof(SiteWord) == 4u, "canonical SiteWord ABI changed");

struct DeviceFactorRing {
  SiteWord* producer[kFixtureContacts]{};
  SiteWord* source = nullptr;
  SiteWord* lane[kHorizon]{};
  SiteWord* expiry[kHorizon]{};
  SiteWord* lesion = nullptr;
};

__host__ __device__ inline SiteWord zero_word() {
  return 0u;
}

__host__ __device__ inline bool is_zero(const SiteWord& word) {
  return word == 0u;
}

constexpr std::int32_t kInvalidAmplitude = -129;

__host__ __device__ inline SiteWord encode_amplitude(std::int32_t value) {
  if (value < -128 || value > 127)
    return zero_word();
  const std::uint8_t face = static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
  const std::uint8_t carrier = static_cast<std::uint8_t>(~face);
  return static_cast<SiteWord>(carrier) |
         (static_cast<SiteWord>(face) << substrate::bcc32::kFaceShift);
}

__host__ __device__ inline bool is_canonical_amplitude(const SiteWord& word) {
  const std::uint8_t carrier = static_cast<std::uint8_t>(substrate::bcc32::carriers(word));
  const std::uint8_t face = static_cast<std::uint8_t>(substrate::bcc32::faces(word));
  const SiteWord other = word & ~(substrate::bcc32::kCarrierMask | substrate::bcc32::kFaceMask);
  return other == 0u && static_cast<std::uint8_t>(carrier ^ face) == 0xffu &&
         static_cast<std::uint8_t>(carrier & face) == 0u;
}

__host__ __device__ inline std::int32_t decode_amplitude(const SiteWord& word) {
  if (!is_canonical_amplitude(word))
    return kInvalidAmplitude;
  return static_cast<std::int32_t>(static_cast<std::int8_t>(substrate::bcc32::faces(word)));
}

__host__ __device__ inline std::uint32_t popcount8(std::uint8_t value) {
  std::uint32_t count = 0u;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit)
    count += (value >> bit) & 1u;
  return count;
}

__host__ __device__ inline std::uint32_t matter_quanta(const SiteWord& word) {
  return popcount8(static_cast<std::uint8_t>(word)) +
         popcount8(static_cast<std::uint8_t>(word >> 8u)) +
         popcount8(static_cast<std::uint8_t>(word >> 16u)) +
         popcount8(static_cast<std::uint8_t>(word >> 24u));
}

__host__ __device__ inline std::uint64_t pack_word(const SiteWord& word) {
  return static_cast<std::uint64_t>(word);
}

__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

__device__ inline SiteWord* segment(DeviceFactorRing view, std::uint32_t id) {
  if (id < kFixtureContacts)
    return view.producer[id];
  if (id == kFixtureContacts)
    return view.source;
  if (id < kFixtureContacts + kHorizon + 1u)
    return view.lane[id - kFixtureContacts - 1u];
  if (id < kFixtureContacts + 2u * kHorizon + 1u)
    return view.expiry[id - kFixtureContacts - kHorizon - 1u];
  return view.lesion;
}

__device__ inline const SiteWord* segment_const(DeviceFactorRing view, std::uint32_t id) {
  return segment(view, id);
}

__host__ __device__ inline std::uint32_t family_offset(std::uint32_t family) {
  if (family < kFactorFamilies)
    return family * kFactorWords;
  if (family == kFactorFamilies)
    return kFactorFamilies * kFactorWords;
  return kFactorFamilies * kFactorWords + kInputWords;
}

__host__ __device__ inline std::uint32_t family_words(std::uint32_t family) {
  if (family < kFactorFamilies)
    return kFactorWords;
  if (family == kFactorFamilies)
    return kInputWords;
  return kPreviousWords;
}

__device__ inline std::int32_t seeded_amplitude(std::uint32_t contact, std::uint32_t index) {
  const std::uint32_t code =
      (index * 37u + contact * 71u + (index >> 5u) * 11u + 19u) & 0xffu;
  return static_cast<std::int32_t>(code) - 128;
}

static __global__ void seed_population_kernel(SiteWord* producer, std::uint32_t contact) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (producer == nullptr || index >= kPopulationWords)
    return;
  producer[index] = encode_amplitude(seeded_amplitude(contact, index));
}

static __global__ void encode_domain_kernel(SiteWord* words, std::int32_t* decoded) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 256u)
    return;
  const std::int32_t value = static_cast<std::int32_t>(index) - 128;
  const SiteWord word = encode_amplitude(value);
  words[index] = word;
  decoded[index] = decode_amplitude(word);
}

static __global__ void produce_into_source_kernel(SiteWord* producer, SiteWord* source) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (producer == nullptr || source == nullptr || index >= kPopulationWords)
    return;
  const SiteWord temporary = producer[index];
  producer[index] = source[index];
  source[index] = temporary;
}

static __global__ void reverse_source_production_kernel(SiteWord* producer, SiteWord* source) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (producer == nullptr || source == nullptr || index >= kPopulationWords)
    return;
  const SiteWord temporary = producer[index];
  producer[index] = source[index];
  source[index] = temporary;
}

static __global__ void arm_next_zero_lane_kernel(DeviceFactorRing view, std::uint32_t contact) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (view.source == nullptr || index >= kPopulationWords)
    return;
  SiteWord* const next = view.lane[contact % kHorizon];
  if (next == nullptr)
    return;
  const SiteWord temporary = view.source[index];
  view.source[index] = next[index];
  next[index] = temporary;
}

static __global__ void reverse_latest_arm_kernel(DeviceFactorRing view, std::uint32_t contact) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (view.source == nullptr || index >= kPopulationWords)
    return;
  SiteWord* const lane = view.lane[contact % kHorizon];
  if (lane == nullptr)
    return;
  const SiteWord temporary = view.source[index];
  view.source[index] = lane[index];
  lane[index] = temporary;
}

static __device__ __forceinline__ void swap_lane_and_expiry(
    DeviceFactorRing view, std::uint32_t contact) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (contact < kHorizon || index >= kPopulationWords)
    return;
  const std::uint32_t lane = (contact - kHorizon) % kHorizon;
  SiteWord* const expired = view.lane[lane];
  SiteWord* const expiry = view.expiry[lane];
  if (expired == nullptr || expiry == nullptr)
    return;
  const SiteWord temporary = expired[index];
  expired[index] = expiry[index];
  expiry[index] = temporary;
}

// A pure lane<->expiry swap is its own inverse; the two kernels below stay
// distinct launch targets for the reversibility contract, not because their
// bodies differ.
static __global__ void expire_age_boundary_kernel(DeviceFactorRing view, std::uint32_t contact) {
  swap_lane_and_expiry(view, contact);
}

static __global__ void reverse_expiry_kernel(DeviceFactorRing view, std::uint32_t contact) {
  swap_lane_and_expiry(view, contact);
}

static __global__ void lesion_family_kernel(SiteWord* words, SiteWord* lesion,
                                            std::uint32_t family) {
  const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (words == nullptr || lesion == nullptr || family >= kFamilyCount ||
      local >= family_words(family))
    return;
  const std::uint32_t index = family_offset(family) + local;
  const SiteWord temporary = words[index];
  words[index] = lesion[index];
  lesion[index] = temporary;
}

static __global__ void clear_state_kernel(DeviceFactorRing view) {
  const std::uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total = kStateSegments * kPopulationWords;
  if (flat >= total)
    return;
  segment(view, flat / kPopulationWords)[flat % kPopulationWords] = zero_word();
}

static __global__ void copy_state_kernel(DeviceFactorRing destination, DeviceFactorRing source) {
  const std::uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total = kStateSegments * kPopulationWords;
  if (flat >= total)
    return;
  const std::uint32_t id = flat / kPopulationWords;
  const std::uint32_t index = flat % kPopulationWords;
  segment(destination, id)[index] = segment_const(source, id)[index];
}

static __global__ void state_hash_kernel(DeviceFactorRing view, std::uint64_t* output) {
  extern __shared__ std::uint64_t shared[];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  const std::uint32_t total = kStateSegments * kPopulationWords;
  std::uint64_t local = 0ull;
  for (std::uint32_t flat = blockIdx.x * blockDim.x + tid; flat < total; flat += stride) {
    const std::uint32_t id = flat / kPopulationWords;
    const std::uint64_t raw = pack_word(segment_const(view, id)[flat % kPopulationWords]);
    if (raw != 0ull)
      local ^= mix64(raw ^ (static_cast<std::uint64_t>(flat) << 1u));
  }
  shared[tid] = local;
  __syncthreads();
  for (std::uint32_t span = blockDim.x >> 1u; span != 0u; span >>= 1u) {
    if (tid < span)
      shared[tid] ^= shared[tid + span];
    __syncthreads();
  }
  if (tid == 0u)
    atomicXor(reinterpret_cast<unsigned long long*>(output), shared[0]);
}

static __global__ void segment_hash_kernel(const SiteWord* words, std::uint64_t* output) {
  extern __shared__ std::uint64_t shared[];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  std::uint64_t local = 0ull;
  for (std::uint32_t index = blockIdx.x * blockDim.x + tid; index < kPopulationWords;
       index += stride) {
    const std::uint64_t raw = pack_word(words[index]);
    if (raw != 0ull)
      local ^= mix64(raw ^ (static_cast<std::uint64_t>(index) << 1u));
  }
  shared[tid] = local;
  __syncthreads();
  for (std::uint32_t span = blockDim.x >> 1u; span != 0u; span >>= 1u) {
    if (tid < span)
      shared[tid] ^= shared[tid + span];
    __syncthreads();
  }
  if (tid == 0u)
    atomicXor(reinterpret_cast<unsigned long long*>(output), shared[0]);
}

static __global__ void state_matter_kernel(DeviceFactorRing view, std::uint64_t* output) {
  extern __shared__ std::uint64_t shared[];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  const std::uint32_t total = kStateSegments * kPopulationWords;
  std::uint64_t local = 0ull;
  for (std::uint32_t flat = blockIdx.x * blockDim.x + tid; flat < total; flat += stride) {
    const std::uint32_t id = flat / kPopulationWords;
    local += matter_quanta(segment_const(view, id)[flat % kPopulationWords]);
  }
  shared[tid] = local;
  __syncthreads();
  for (std::uint32_t span = blockDim.x >> 1u; span != 0u; span >>= 1u) {
    if (tid < span)
      shared[tid] += shared[tid + span];
    __syncthreads();
  }
  if (tid == 0u)
    atomicAdd(reinterpret_cast<unsigned long long*>(output), shared[0]);
}

static __global__ void segment_matter_kernel(const SiteWord* words, std::uint64_t* output) {
  extern __shared__ std::uint64_t shared[];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  std::uint64_t local = 0ull;
  for (std::uint32_t index = blockIdx.x * blockDim.x + tid; index < kPopulationWords;
       index += stride)
    local += matter_quanta(words[index]);
  shared[tid] = local;
  __syncthreads();
  for (std::uint32_t span = blockDim.x >> 1u; span != 0u; span >>= 1u) {
    if (tid < span)
      shared[tid] += shared[tid + span];
    __syncthreads();
  }
  if (tid == 0u)
    atomicAdd(reinterpret_cast<unsigned long long*>(output), shared[0]);
}

static __global__ void state_mismatch_kernel(DeviceFactorRing left, DeviceFactorRing right,
                                             std::uint32_t* output) {
  const std::uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total = kStateSegments * kPopulationWords;
  if (flat >= total)
    return;
  const std::uint32_t id = flat / kPopulationWords;
  const std::uint32_t index = flat % kPopulationWords;
  if (pack_word(segment_const(left, id)[index]) != pack_word(segment_const(right, id)[index]))
    atomicAdd(output, 1u);
}

static __global__ void segment_mismatch_kernel(const SiteWord* left, const SiteWord* right,
                                               std::uint32_t count, std::uint32_t* output) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (left == nullptr || right == nullptr || output == nullptr || index >= count)
    return;
  if (left[index] != right[index])
    atomicAdd(output, 1u);
}

static __global__ void amplitude_count_kernel(const SiteWord* words, std::uint32_t count,
                                              std::int32_t value,
                                              std::uint32_t* output) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (words == nullptr || output == nullptr || index >= count)
    return;
  const SiteWord word = words[index];
  if (is_canonical_amplitude(word) && decode_amplitude(word) == value)
    atomicAdd(output, 1u);
}

}  // namespace bcc32_b3_factor_amplitude_matter
