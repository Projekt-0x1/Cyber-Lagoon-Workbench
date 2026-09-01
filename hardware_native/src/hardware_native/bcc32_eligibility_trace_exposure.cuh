#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_eligibility_trace_exposure.hpp"

namespace substrate::bcc32 {
namespace carrier_vacancy_exposure_detail {

static __global__ void dense_kernel(SiteWord* words,
                                    std::uint64_t word_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < word_count)
    words[index] = carrier_vacancy_exposure_word(words[index]);
}

static __global__ void active_kernel(SiteWord* words,
                                     std::uint64_t word_count,
                                     const std::uint64_t* active_slots,
                                     std::uint64_t active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= active_count)
    return;
  const std::uint64_t slot = active_slots[index];
  if (slot < word_count && carrier_vacancy_active_slot_is_unique(
                              active_slots, active_count, index))
    words[slot] = carrier_vacancy_exposure_word(words[slot]);
}

}  // namespace carrier_vacancy_exposure_detail

inline void apply_cuda_carrier_vacancy_exposure_dense(
    SiteWord* words, std::uint64_t word_count, cudaStream_t stream = nullptr) {
  if (words == nullptr || word_count == 0u)
    return;
  constexpr std::uint32_t kThreads = 256u;
  const std::uint64_t block_count =
      (word_count + kThreads - 1u) / kThreads;
  carrier_vacancy_exposure_detail::dense_kernel<<<
      static_cast<unsigned int>(block_count), kThreads, 0u, stream>>>(
      words, word_count);
}

inline void apply_cuda_carrier_vacancy_exposure_active(
    SiteWord* words, std::uint64_t word_count,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream = nullptr) {
  if (words == nullptr || active_slots == nullptr || active_count == 0u)
    return;
  constexpr std::uint32_t kThreads = 256u;
  const std::uint64_t block_count =
      (active_count + kThreads - 1u) / kThreads;
  carrier_vacancy_exposure_detail::active_kernel<<<
      static_cast<unsigned int>(block_count), kThreads, 0u, stream>>>(
      words, word_count, active_slots, active_count);
}

}  // namespace substrate::bcc32
