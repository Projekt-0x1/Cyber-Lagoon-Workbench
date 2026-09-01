#pragma once

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace substrate::bcc32::resident_recurrent_material {

inline constexpr std::uint32_t kCoefficientCount = 384u * 32u;
inline constexpr std::uint32_t kCohortCount = 2u;
inline constexpr std::uint32_t kWordsPerSign = kCoefficientCount * kCohortCount;
inline constexpr std::uint32_t kAdapterBudget = kCoefficientCount * 64u;
inline constexpr std::uint32_t kFreeSlabWords = (kAdapterBudget + 31u) / 32u;
inline constexpr std::uint64_t kTotalQuanta = kAdapterBudget;

enum class PublicationStatus : std::uint32_t {
  kNoop = 0u,
  kCommitted = 1u,
  kRejected = 2u,
};

enum class PublicationReason : std::uint32_t {
  kNone = 0u,
  kInvalidTarget = 1u,
  kJournalActive = 2u,
  kLesionActive = 3u,
  kInsufficientFree = 4u,
  kInvariant = 5u,
};

struct MaterialScalars {
  std::uint64_t total = 0u;
  std::uint64_t active = 0u;
  std::uint64_t free = 0u;
  std::uint64_t escrow = 0u;
};

struct JournalMetadata {
  std::uint32_t active = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t added = 0u;
  std::uint64_t removed = 0u;
  std::uint64_t active_before = 0u;
  std::uint64_t free_before = 0u;
};

struct LesionMetadata {
  std::uint32_t active = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t active_before = 0u;
  std::uint64_t free_before = 0u;
  std::uint64_t escrow_before = 0u;
};

struct DeviceView {
  std::uint32_t* resident_positive = nullptr;
  std::uint32_t* resident_negative = nullptr;
  std::uint32_t* free_slab = nullptr;
  MaterialScalars* scalars = nullptr;
  JournalMetadata* journal = nullptr;
  std::uint32_t* journal_positive = nullptr;
  std::uint32_t* journal_negative = nullptr;
  LesionMetadata* lesion = nullptr;
  std::uint32_t* lesion_positive = nullptr;
  std::uint32_t* lesion_negative = nullptr;
};

struct DevicePublicationReceipt {
  std::uint32_t status = 0u;
  std::uint32_t reason = 0u;
  std::uint32_t invalid_target = 0u;
  std::uint32_t device_reads = 0u;
  std::uint64_t added = 0u;
  std::uint64_t removed = 0u;
  std::uint64_t total = 0u;
  std::uint64_t active_before = 0u;
  std::uint64_t active_after = 0u;
  std::uint64_t free_before = 0u;
  std::uint64_t free_after = 0u;
  std::uint64_t escrow_before = 0u;
  std::uint64_t escrow_after = 0u;
};

struct DeviceInitializationReceipt {
  std::uint32_t invalid = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t active = 0u;
};

struct PublicationReceipt {
  PublicationStatus status = PublicationStatus::kRejected;
  PublicationReason reason = PublicationReason::kInvariant;
  std::uint32_t invalid_target = 0u;
  std::uint32_t device_reads = 0u;
  std::uint64_t added = 0u;
  std::uint64_t removed = 0u;
  std::uint64_t total = 0u;
  std::uint64_t active_before = 0u;
  std::uint64_t active_after = 0u;
  std::uint64_t free_before = 0u;
  std::uint64_t free_after = 0u;
  std::uint64_t escrow_before = 0u;
  std::uint64_t escrow_after = 0u;
};

struct Inspection {
  MaterialScalars scalars{};
  std::uint64_t active_positive = 0u;
  std::uint64_t active_negative = 0u;
  std::uint64_t free_slab = 0u;
  std::uint64_t lesion_positive = 0u;
  std::uint64_t lesion_negative = 0u;

  std::uint64_t conserved_quanta() const {
    return active_positive + active_negative + free_slab +
           lesion_positive + lesion_negative;
  }
};

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

__device__ inline std::uint64_t popcount_words(const std::uint32_t* words) {
  std::uint64_t count = 0u;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x)
    count += __popc(words[index]);
  return count;
}

__device__ inline std::uint32_t prefix_word(std::uint32_t index,
                                            std::uint64_t quanta) {
  const std::uint64_t first = static_cast<std::uint64_t>(index) * 32u;
  if (quanta <= first) return 0u;
  if (quanta >= first + 32u) return 0xffffffffu;
  return (1u << static_cast<std::uint32_t>(quanta - first)) - 1u;
}

__device__ inline void write_prefix(std::uint32_t* slab,
                                    std::uint64_t quanta) {
  for (std::uint32_t index = threadIdx.x; index < kFreeSlabWords;
       index += blockDim.x)
    slab[index] = prefix_word(index, quanta);
}

__device__ inline bool canonical_prefix(std::uint32_t low, std::uint32_t high) {
  if (high != 0u) {
    // The predecessor exporter reserved one bit in each 32-bit cohort.
    // Accept that exact 31+31 unary form when adopting existing matter.
    if (low != 0xffffffffu && low != 0x7fffffffu) return false;
    if (high == 0xffffffffu) return true;
    const std::uint32_t bit = 31u - __clz(high);
    const std::uint32_t expected =
        bit == 31u ? 0xffffffffu : ((1u << (bit + 1u)) - 1u);
    return high == expected;
  }
  if (low == 0u) return true;
  const std::uint32_t bit = 31u - __clz(low);
  return low == (bit == 31u ? 0xffffffffu : ((1u << (bit + 1u)) - 1u));
}

__global__ inline void initialize_material_kernel(
    const std::uint32_t* positive, const std::uint32_t* negative,
    DeviceView material, DeviceInitializationReceipt* receipt) {
  __shared__ std::uint64_t active;
  __shared__ std::uint32_t invalid;
  if (threadIdx.x == 0u) {
    material.scalars->total = kTotalQuanta;
    material.scalars->active = 0u;
    material.scalars->free = 0u;
    material.scalars->escrow = 0u;
    material.journal->active = 0u;
    material.journal->reserved = 0u;
    material.journal->added = 0u;
    material.journal->removed = 0u;
    material.journal->active_before = 0u;
    material.journal->free_before = 0u;
    material.lesion->active = 0u;
    material.lesion->reserved = 0u;
    material.lesion->active_before = 0u;
    material.lesion->free_before = 0u;
    material.lesion->escrow_before = 0u;
    active = 0u;
    invalid = 0u;
  }
  __syncthreads();
  std::uint64_t local = popcount_words(positive) + popcount_words(negative);
  atomicAdd(reinterpret_cast<unsigned long long*>(&active),
            static_cast<unsigned long long>(local));
  __syncthreads();
  for (std::uint32_t coefficient = threadIdx.x;
       coefficient < kCoefficientCount; coefficient += blockDim.x) {
    const std::uint64_t offset = static_cast<std::uint64_t>(coefficient) * 2u;
    const std::uint32_t positive_low = positive[offset];
    const std::uint32_t positive_high = positive[offset + 1u];
    const std::uint32_t negative_low = negative[offset];
    const std::uint32_t negative_high = negative[offset + 1u];
    if (!canonical_prefix(positive_low, positive_high) ||
        !canonical_prefix(negative_low, negative_high) ||
        ((positive_low | positive_high) != 0u &&
         (negative_low | negative_high) != 0u))
      atomicOr(&invalid, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    if (active > kTotalQuanta) invalid = 1u;
    material.scalars->active = active;
    material.scalars->free = invalid == 0u ? kTotalQuanta - active : 0u;
    receipt->invalid = invalid;
    receipt->active = active;
  }
  __syncthreads();
  write_prefix(material.free_slab, material.scalars->free);
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    material.journal_positive[index] = 0u;
    material.journal_negative[index] = 0u;
    material.lesion_positive[index] = 0u;
    material.lesion_negative[index] = 0u;
  }
}

__global__ inline void publish_material_kernel(
    const std::uint32_t* target_positive, const std::uint32_t* target_negative,
    DeviceView material, DevicePublicationReceipt* receipt) {
  __shared__ std::uint64_t added;
  __shared__ std::uint64_t removed;
  __shared__ std::uint64_t current_active;
  __shared__ std::uint64_t current_free;
  __shared__ std::uint64_t current_escrow;
  __shared__ std::uint32_t target_invalid;
  __shared__ std::uint32_t current_invalid;
  __shared__ std::uint32_t free_invalid;
  __shared__ std::uint32_t same;
  __shared__ std::uint32_t status;
  if (threadIdx.x == 0u) {
    added = 0u;
    removed = 0u;
    current_active = 0u;
    current_free = 0u;
    current_escrow = 0u;
    target_invalid = 0u;
    current_invalid = 0u;
    free_invalid = 0u;
    same = 1u;
    status = static_cast<std::uint32_t>(PublicationStatus::kRejected);
  }
  __syncthreads();
  std::uint64_t local_added = 0u;
  std::uint64_t local_removed = 0u;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    const std::uint32_t old_positive = material.resident_positive[index];
    const std::uint32_t old_negative = material.resident_negative[index];
    const std::uint32_t new_positive = target_positive[index];
    const std::uint32_t new_negative = target_negative[index];
    if (old_positive != new_positive || old_negative != new_negative)
      atomicExch(&same, 0u);
    local_added += __popc(new_positive & ~old_positive);
    local_added += __popc(new_negative & ~old_negative);
    local_removed += __popc(old_positive & ~new_positive);
    local_removed += __popc(old_negative & ~new_negative);
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&added),
            static_cast<unsigned long long>(local_added));
  atomicAdd(reinterpret_cast<unsigned long long*>(&removed),
            static_cast<unsigned long long>(local_removed));
  std::uint64_t local_active = 0u;
  std::uint64_t local_free = 0u;
  std::uint64_t local_escrow = 0u;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    local_active += __popc(material.resident_positive[index]);
    local_active += __popc(material.resident_negative[index]);
    local_escrow += __popc(material.lesion_positive[index]);
    local_escrow += __popc(material.lesion_negative[index]);
  }
  for (std::uint32_t index = threadIdx.x; index < kFreeSlabWords;
       index += blockDim.x) {
    if (material.free_slab[index] !=
        prefix_word(index, material.scalars->free))
      atomicOr(&free_invalid, 1u);
    else
      local_free += __popc(material.free_slab[index]);
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&current_active),
            static_cast<unsigned long long>(local_active));
  atomicAdd(reinterpret_cast<unsigned long long*>(&current_free),
            static_cast<unsigned long long>(local_free));
  atomicAdd(reinterpret_cast<unsigned long long*>(&current_escrow),
            static_cast<unsigned long long>(local_escrow));
  for (std::uint32_t coefficient = threadIdx.x;
       coefficient < kCoefficientCount; coefficient += blockDim.x) {
    const std::uint64_t offset = static_cast<std::uint64_t>(coefficient) * 2u;
    const std::uint32_t old_positive_low = material.resident_positive[offset];
    const std::uint32_t old_positive_high = material.resident_positive[offset + 1u];
    const std::uint32_t old_negative_low = material.resident_negative[offset];
    const std::uint32_t old_negative_high = material.resident_negative[offset + 1u];
    const std::uint32_t new_positive_low = target_positive[offset];
    const std::uint32_t new_positive_high = target_positive[offset + 1u];
    const std::uint32_t new_negative_low = target_negative[offset];
    const std::uint32_t new_negative_high = target_negative[offset + 1u];
    if (!canonical_prefix(old_positive_low, old_positive_high) ||
        !canonical_prefix(old_negative_low, old_negative_high) ||
        ((old_positive_low | old_positive_high) != 0u &&
         (old_negative_low | old_negative_high) != 0u))
      atomicOr(&current_invalid, 1u);
    if (!canonical_prefix(new_positive_low, new_positive_high) ||
        !canonical_prefix(new_negative_low, new_negative_high) ||
        ((new_positive_low | new_positive_high) != 0u &&
         (new_negative_low | new_negative_high) != 0u))
      atomicOr(&target_invalid, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const MaterialScalars before = *material.scalars;
    receipt->device_reads = 1u;
    receipt->total = before.total;
    receipt->active_before = before.active;
    receipt->free_before = before.free;
    receipt->escrow_before = before.escrow;
    receipt->added = added;
    receipt->removed = removed;
    receipt->invalid_target = target_invalid;
    const bool invariant = before.total == kTotalQuanta &&
                           current_active == before.active &&
                           current_free == before.free &&
                           current_escrow == before.escrow &&
                           before.active + before.free + before.escrow == before.total;
    if (target_invalid != 0u) {
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kInvalidTarget);
    } else if (current_invalid != 0u || free_invalid != 0u || !invariant) {
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kInvariant);
    } else if (material.lesion->active != 0u) {
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kLesionActive);
    } else if (same != 0u) {
      status = static_cast<std::uint32_t>(PublicationStatus::kNoop);
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kNone);
    } else if (material.journal->active != 0u) {
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kJournalActive);
    } else if (before.free + removed < added) {
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kInsufficientFree);
    } else {
      status = static_cast<std::uint32_t>(PublicationStatus::kCommitted);
      receipt->reason = static_cast<std::uint32_t>(PublicationReason::kNone);
    }
    receipt->status = status;
  }
  __syncthreads();
  if (status == static_cast<std::uint32_t>(PublicationStatus::kCommitted)) {
    for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
         index += blockDim.x) {
      material.journal_positive[index] =
          material.resident_positive[index] ^ target_positive[index];
      material.journal_negative[index] =
          material.resident_negative[index] ^ target_negative[index];
      material.resident_positive[index] = target_positive[index];
      material.resident_negative[index] = target_negative[index];
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      const MaterialScalars before = *material.scalars;
      material.journal->active = 1u;
      material.journal->added = added;
      material.journal->removed = removed;
      material.journal->active_before = before.active;
      material.journal->free_before = before.free;
      material.scalars->active = before.active - removed + added;
      material.scalars->free = before.free + removed - added;
    }
    __syncthreads();
    write_prefix(material.free_slab, material.scalars->free);
    __syncthreads();
    if (threadIdx.x == 0u) {
      receipt->active_after = material.scalars->active;
      receipt->free_after = material.scalars->free;
      receipt->escrow_after = material.scalars->escrow;
    }
  } else {
    __syncthreads();
    if (threadIdx.x == 0u) {
      const MaterialScalars before = *material.scalars;
      receipt->active_after = before.active;
      receipt->free_after = before.free;
      receipt->escrow_after = before.escrow;
    }
  }
}

__global__ inline void inverse_material_kernel(
    DeviceView material, DevicePublicationReceipt* receipt) {
  if (threadIdx.x == 0u) {
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kRejected);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kJournalActive);
    receipt->device_reads = 1u;
    receipt->invalid_target = 0u;
  }
  __syncthreads();
  if (material.journal->active == 0u || material.lesion->active != 0u) return;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    material.resident_positive[index] ^= material.journal_positive[index];
    material.resident_negative[index] ^= material.journal_negative[index];
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    material.scalars->active = material.journal->active_before;
    material.scalars->free = material.journal->free_before;
  }
  __syncthreads();
  write_prefix(material.free_slab, material.scalars->free);
  __syncthreads();
  if (threadIdx.x == 0u) {
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kCommitted);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kNone);
    receipt->active_after = material.scalars->active;
    receipt->free_after = material.scalars->free;
    receipt->escrow_after = material.scalars->escrow;
    material.journal->active = 0u;
    material.journal->added = 0u;
    material.journal->removed = 0u;
    material.journal->active_before = 0u;
    material.journal->free_before = 0u;
  }
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    material.journal_positive[index] = 0u;
    material.journal_negative[index] = 0u;
  }
}

__global__ inline void lesion_material_kernel(DeviceView material,
                                               DevicePublicationReceipt* receipt) {
  if (threadIdx.x == 0u) {
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kRejected);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kLesionActive);
    receipt->device_reads = 1u;
  }
  __syncthreads();
  if (material.lesion->active != 0u || material.scalars->escrow != 0u) return;
  const MaterialScalars before = *material.scalars;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    material.lesion_positive[index] = material.resident_positive[index];
    material.lesion_negative[index] = material.resident_negative[index];
    material.resident_positive[index] = 0u;
    material.resident_negative[index] = 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    material.lesion->active = 1u;
    material.lesion->active_before = before.active;
    material.lesion->free_before = before.free;
    material.lesion->escrow_before = before.escrow;
    material.scalars->active = 0u;
    material.scalars->escrow = before.escrow + before.active;
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kCommitted);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kNone);
    receipt->active_before = before.active;
    receipt->active_after = 0u;
    receipt->free_before = before.free;
    receipt->free_after = before.free;
    receipt->escrow_before = before.escrow;
    receipt->escrow_after = material.scalars->escrow;
  }
}

__global__ inline void restore_lesion_material_kernel(
    DeviceView material, DevicePublicationReceipt* receipt) {
  if (threadIdx.x == 0u) {
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kRejected);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kLesionActive);
    receipt->device_reads = 1u;
  }
  __syncthreads();
  if (material.lesion->active == 0u) return;
  for (std::uint32_t index = threadIdx.x; index < kWordsPerSign;
       index += blockDim.x) {
    material.resident_positive[index] = material.lesion_positive[index];
    material.resident_negative[index] = material.lesion_negative[index];
    material.lesion_positive[index] = 0u;
    material.lesion_negative[index] = 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    material.scalars->active = material.lesion->active_before;
    material.scalars->free = material.lesion->free_before;
    material.scalars->escrow = material.lesion->escrow_before;
    material.lesion->active = 0u;
    material.lesion->active_before = 0u;
    material.lesion->free_before = 0u;
    material.lesion->escrow_before = 0u;
    receipt->status = static_cast<std::uint32_t>(PublicationStatus::kCommitted);
    receipt->reason = static_cast<std::uint32_t>(PublicationReason::kNone);
    receipt->active_after = material.scalars->active;
    receipt->free_after = material.scalars->free;
    receipt->escrow_after = material.scalars->escrow;
  }
}

inline PublicationReceipt copy_publication_receipt(
    DevicePublicationReceipt* device_receipt, const char* label) {
  DevicePublicationReceipt device{};
  require_cuda(cudaMemcpy(&device, device_receipt, sizeof(device),
                          cudaMemcpyDeviceToHost),
               label);
  (void)cudaFree(device_receipt);
  return {static_cast<PublicationStatus>(device.status),
          static_cast<PublicationReason>(device.reason), device.invalid_target,
          device.device_reads, device.added, device.removed, device.total,
          device.active_before, device.active_after, device.free_before,
          device.free_after, device.escrow_before, device.escrow_after};
}

inline PublicationReceipt read_publication_receipt(
    const DevicePublicationReceipt* device_receipt, const char* label) {
  DevicePublicationReceipt device{};
  require_cuda(cudaMemcpy(&device, device_receipt, sizeof(device),
                          cudaMemcpyDeviceToHost),
               label);
  return {static_cast<PublicationStatus>(device.status),
          static_cast<PublicationReason>(device.reason), device.invalid_target,
          device.device_reads, device.added, device.removed, device.total,
          device.active_before, device.active_after, device.free_before,
          device.free_after, device.escrow_before, device.escrow_after};
}

inline void publish_async(DeviceView view, const std::uint32_t* positive,
                          const std::uint32_t* negative,
                          DevicePublicationReceipt* receipt) {
  if (receipt == nullptr)
    throw std::runtime_error("adapter publication receipt is null");
  require_cuda(cudaMemsetAsync(receipt, 0, sizeof(*receipt), 0),
               "clear adapter publication receipt");
  publish_material_kernel<<<1u, 256u, 0u, 0u>>>(positive, negative, view,
                                                 receipt);
  require_cuda(cudaGetLastError(), "launch adapter async publication");
}

inline PublicationReceipt publish(DeviceView view, const std::uint32_t* positive,
                                  const std::uint32_t* negative) {
  DevicePublicationReceipt* receipt = nullptr;
  require_cuda(cudaMalloc(reinterpret_cast<void**>(&receipt), sizeof(*receipt)),
               "allocate adapter publication receipt");
  try {
    publish_async(view, positive, negative, receipt);
    // 0X1-284: no explicit sync needed -- the kernel above and the
    // synchronous cudaMemcpy inside copy_publication_receipt() both run on
    // the legacy default stream, so the D2H copy already blocks the host
    // until this kernel completes.
    return copy_publication_receipt(receipt, "read adapter publication receipt");
  } catch (...) {
    (void)cudaFree(receipt);
    throw;
  }
}

inline PublicationReceipt inverse(DeviceView view) {
  DevicePublicationReceipt* receipt = nullptr;
  require_cuda(cudaMalloc(reinterpret_cast<void**>(&receipt), sizeof(*receipt)),
               "allocate adapter inverse receipt");
  try {
    require_cuda(cudaMemset(receipt, 0, sizeof(*receipt)),
                 "clear adapter inverse receipt");
    inverse_material_kernel<<<1u, 256u>>>(view, receipt);
    require_cuda(cudaGetLastError(), "launch adapter inverse");
    // 0X1-284: redundant -- copy_publication_receipt()'s synchronous D2H
    // cudaMemcpy is on the same legacy default stream as the kernel above
    // and already blocks the host until it completes.
    return copy_publication_receipt(receipt, "read adapter inverse receipt");
  } catch (...) {
    (void)cudaFree(receipt);
    throw;
  }
}

inline PublicationReceipt lesion(DeviceView view) {
  DevicePublicationReceipt* receipt = nullptr;
  require_cuda(cudaMalloc(reinterpret_cast<void**>(&receipt), sizeof(*receipt)),
               "allocate adapter lesion receipt");
  try {
    require_cuda(cudaMemset(receipt, 0, sizeof(*receipt)),
                 "clear adapter lesion receipt");
    lesion_material_kernel<<<1u, 256u>>>(view, receipt);
    require_cuda(cudaGetLastError(), "launch adapter lesion");
    // 0X1-284: redundant -- see adapter-inverse note above; same stream,
    // same synchronous D2H memcpy immediately follows.
    return copy_publication_receipt(receipt, "read adapter lesion receipt");
  } catch (...) {
    (void)cudaFree(receipt);
    throw;
  }
}

inline PublicationReceipt restore_lesion(DeviceView view) {
  DevicePublicationReceipt* receipt = nullptr;
  require_cuda(cudaMalloc(reinterpret_cast<void**>(&receipt), sizeof(*receipt)),
               "allocate adapter lesion restore receipt");
  try {
    require_cuda(cudaMemset(receipt, 0, sizeof(*receipt)),
                 "clear adapter lesion restore receipt");
    restore_lesion_material_kernel<<<1u, 256u>>>(view, receipt);
    require_cuda(cudaGetLastError(), "launch adapter lesion restore");
    // 0X1-284: redundant -- see adapter-inverse note above; same stream,
    // same synchronous D2H memcpy immediately follows.
    return copy_publication_receipt(receipt, "read adapter lesion restore receipt");
  } catch (...) {
    (void)cudaFree(receipt);
    throw;
  }
}

class ResidentMaterial {
 public:
  ResidentMaterial() = default;
  ResidentMaterial(const ResidentMaterial&) = delete;
  ResidentMaterial& operator=(const ResidentMaterial&) = delete;
  ResidentMaterial(ResidentMaterial&& other) noexcept { move_from(other); }
  ResidentMaterial& operator=(ResidentMaterial&& other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }
  ~ResidentMaterial() { release(); }

  void initialize_from_device(const std::uint32_t* positive,
                              const std::uint32_t* negative) {
    release();
    allocate();
    DeviceInitializationReceipt* device_receipt = nullptr;
    try {
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&device_receipt),
                              sizeof(*device_receipt)),
                   "allocate adapter material initialization receipt");
      require_cuda(cudaMemset(device_receipt, 0, sizeof(*device_receipt)),
                   "clear adapter material initialization receipt");
      initialize_material_kernel<<<1u, 256u>>>(positive, negative,
                                                view(nullptr, nullptr),
                                                device_receipt);
      require_cuda(cudaGetLastError(), "launch adapter material initialization");
      // 0X1-284: redundant -- the synchronous D2H cudaMemcpy right below is
      // on the same legacy default stream as the kernel above and already
      // blocks the host until the kernel completes.
      DeviceInitializationReceipt host_receipt{};
      require_cuda(cudaMemcpy(&host_receipt, device_receipt,
                              sizeof(host_receipt), cudaMemcpyDeviceToHost),
                   "read adapter material initialization receipt");
      (void)cudaFree(device_receipt);
      device_receipt = nullptr;
      if (host_receipt.invalid != 0u || host_receipt.active > kTotalQuanta)
        throw std::runtime_error("adapter A is not canonical or exceeds material budget");
    } catch (...) {
      if (device_receipt != nullptr) (void)cudaFree(device_receipt);
      release();
      throw;
    }
  }

  DeviceView view(std::uint32_t* positive, std::uint32_t* negative) const {
    return {positive, negative, free_slab_, scalars_, journal_, journal_positive_,
            journal_negative_, lesion_, lesion_positive_,
            lesion_negative_};
  }

  ResidentMaterial snapshot() const {
    ResidentMaterial copy;
    copy.allocate();
    copy_device(copy);
    return copy;
  }

  Inspection inspect(const std::uint32_t* resident_positive,
                     const std::uint32_t* resident_negative) const {
    Inspection result{};
    require_cuda(cudaMemcpy(&result.scalars, scalars_, sizeof(MaterialScalars),
                            cudaMemcpyDeviceToHost),
                 "read adapter inspection scalars");
    const std::size_t words_bytes = static_cast<std::size_t>(kWordsPerSign) * sizeof(std::uint32_t);
    const std::size_t slab_bytes = static_cast<std::size_t>(kFreeSlabWords) * sizeof(std::uint32_t);
    std::vector<std::uint32_t> words(kWordsPerSign);
    require_cuda(cudaMemcpy(words.data(), resident_positive, words_bytes, cudaMemcpyDeviceToHost), "inspect adapter positive");
    for (const std::uint32_t word : words) result.active_positive += __builtin_popcount(word);
    require_cuda(cudaMemcpy(words.data(), resident_negative, words_bytes, cudaMemcpyDeviceToHost), "inspect adapter negative");
    for (const std::uint32_t word : words) result.active_negative += __builtin_popcount(word);
    words.resize(kFreeSlabWords);
    require_cuda(cudaMemcpy(words.data(), free_slab_, slab_bytes, cudaMemcpyDeviceToHost), "inspect adapter free slab");
    for (const std::uint32_t word : words) result.free_slab += __builtin_popcount(word);
    words.resize(kWordsPerSign);
    require_cuda(cudaMemcpy(words.data(), lesion_positive_, words_bytes, cudaMemcpyDeviceToHost), "inspect adapter lesion positive");
    for (const std::uint32_t word : words) result.lesion_positive += __builtin_popcount(word);
    require_cuda(cudaMemcpy(words.data(), lesion_negative_, words_bytes, cudaMemcpyDeviceToHost), "inspect adapter lesion negative");
    for (const std::uint32_t word : words) result.lesion_negative += __builtin_popcount(word);
    return result;
  }

  void restore(const ResidentMaterial& snapshot) {
    if (!allocated() || !snapshot.allocated())
      throw std::runtime_error("adapter material snapshot is unallocated");
    const std::size_t words_bytes = static_cast<std::size_t>(kWordsPerSign) * sizeof(std::uint32_t);
    const std::size_t slab_bytes = static_cast<std::size_t>(kFreeSlabWords) * sizeof(std::uint32_t);
    require_cuda(cudaMemcpy(free_slab_, snapshot.free_slab_, slab_bytes, cudaMemcpyDeviceToDevice), "restore adapter free slab");
    require_cuda(cudaMemcpy(journal_positive_, snapshot.journal_positive_, words_bytes, cudaMemcpyDeviceToDevice), "restore adapter journal positive");
    require_cuda(cudaMemcpy(journal_negative_, snapshot.journal_negative_, words_bytes, cudaMemcpyDeviceToDevice), "restore adapter journal negative");
    require_cuda(cudaMemcpy(lesion_positive_, snapshot.lesion_positive_, words_bytes, cudaMemcpyDeviceToDevice), "restore adapter lesion positive");
    require_cuda(cudaMemcpy(lesion_negative_, snapshot.lesion_negative_, words_bytes, cudaMemcpyDeviceToDevice), "restore adapter lesion negative");
    require_cuda(cudaMemcpy(scalars_, snapshot.scalars_, sizeof(MaterialScalars), cudaMemcpyDeviceToDevice), "restore adapter material scalars");
    require_cuda(cudaMemcpy(journal_, snapshot.journal_, sizeof(JournalMetadata), cudaMemcpyDeviceToDevice), "restore adapter journal");
    require_cuda(cudaMemcpy(lesion_, snapshot.lesion_, sizeof(LesionMetadata), cudaMemcpyDeviceToDevice), "restore adapter lesion metadata");
  }

  void append_hash(std::uint64_t& hash) const {
    MaterialScalars scalars{};
    JournalMetadata journal{};
    LesionMetadata lesion{};
    require_cuda(cudaMemcpy(&scalars, scalars_, sizeof(scalars), cudaMemcpyDeviceToHost), "read adapter material scalars");
    require_cuda(cudaMemcpy(&journal, journal_, sizeof(journal), cudaMemcpyDeviceToHost), "read adapter journal");
    require_cuda(cudaMemcpy(&lesion, lesion_, sizeof(lesion), cudaMemcpyDeviceToHost), "read adapter lesion");
    const std::array<std::uint64_t, 16> metadata{
        scalars.total, scalars.active, scalars.free, scalars.escrow,
        journal.active, journal.added, journal.removed, journal.active_before,
        journal.free_before, lesion.active, lesion.active_before, lesion.free_before,
        lesion.escrow_before, journal.reserved | (static_cast<std::uint64_t>(lesion.reserved) << 32u)};
    for (const std::uint64_t value : metadata) hash_word(hash, value);
    const std::size_t words_bytes = static_cast<std::size_t>(kWordsPerSign) * sizeof(std::uint32_t);
    const std::size_t slab_bytes = static_cast<std::size_t>(kFreeSlabWords) * sizeof(std::uint32_t);
    std::vector<std::uint32_t> words(kWordsPerSign);
    require_cuda(cudaMemcpy(words.data(), free_slab_, slab_bytes, cudaMemcpyDeviceToHost), "hash adapter free slab");
    for (std::uint32_t index = 0u; index < kFreeSlabWords; ++index) hash_word(hash, words[index]);
    require_cuda(cudaMemcpy(words.data(), journal_positive_, words_bytes, cudaMemcpyDeviceToHost), "hash adapter journal positive");
    for (const std::uint32_t word : words) hash_word(hash, word);
    require_cuda(cudaMemcpy(words.data(), journal_negative_, words_bytes, cudaMemcpyDeviceToHost), "hash adapter journal negative");
    for (const std::uint32_t word : words) hash_word(hash, word);
    require_cuda(cudaMemcpy(words.data(), lesion_positive_, words_bytes, cudaMemcpyDeviceToHost), "hash adapter lesion positive");
    for (const std::uint32_t word : words) hash_word(hash, word);
    require_cuda(cudaMemcpy(words.data(), lesion_negative_, words_bytes, cudaMemcpyDeviceToHost), "hash adapter lesion negative");
    for (const std::uint32_t word : words) hash_word(hash, word);
  }

  void release() noexcept {
    free_device(lesion_negative_); free_device(lesion_positive_); free_device(lesion_);
    free_device(journal_negative_); free_device(journal_positive_);
    free_device(journal_); free_device(scalars_); free_device(free_slab_);
  }

 private:
  static void hash_word(std::uint64_t& hash, std::uint64_t value) {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
      hash ^= (value >> shift) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  static void hash_word(std::uint64_t& hash, std::uint32_t value) {
    hash_word(hash, static_cast<std::uint64_t>(value));
  }
  static void free_device(void*& pointer) noexcept { if (pointer != nullptr) (void)cudaFree(pointer); pointer = nullptr; }
  template <typename T> static void free_device(T*& pointer) noexcept { void* raw = pointer; free_device(raw); pointer = nullptr; }
  static void alloc(void** pointer, std::size_t bytes, const char* label) { require_cuda(cudaMalloc(pointer, bytes), label); }
  bool allocated() const { return scalars_ != nullptr; }
  void allocate() {
    const std::size_t words_bytes = static_cast<std::size_t>(kWordsPerSign) * sizeof(std::uint32_t);
    try {
      alloc(reinterpret_cast<void**>(&free_slab_), static_cast<std::size_t>(kFreeSlabWords) * sizeof(std::uint32_t), "allocate adapter free slab");
      alloc(reinterpret_cast<void**>(&scalars_), sizeof(MaterialScalars), "allocate adapter material scalars");
      alloc(reinterpret_cast<void**>(&journal_), sizeof(JournalMetadata), "allocate adapter journal");
      alloc(reinterpret_cast<void**>(&journal_positive_), words_bytes, "allocate adapter journal positive");
      alloc(reinterpret_cast<void**>(&journal_negative_), words_bytes, "allocate adapter journal negative");
      alloc(reinterpret_cast<void**>(&lesion_), sizeof(LesionMetadata), "allocate adapter lesion");
      alloc(reinterpret_cast<void**>(&lesion_positive_), words_bytes, "allocate adapter lesion positive");
      alloc(reinterpret_cast<void**>(&lesion_negative_), words_bytes, "allocate adapter lesion negative");
    } catch (...) { release(); throw; }
  }
  void copy_device(ResidentMaterial& copy) const {
    const std::size_t words_bytes = static_cast<std::size_t>(kWordsPerSign) * sizeof(std::uint32_t);
    require_cuda(cudaMemcpy(copy.free_slab_, free_slab_, static_cast<std::size_t>(kFreeSlabWords) * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice), "snapshot adapter free slab");
    require_cuda(cudaMemcpy(copy.scalars_, scalars_, sizeof(MaterialScalars), cudaMemcpyDeviceToDevice), "snapshot adapter scalars");
    require_cuda(cudaMemcpy(copy.journal_, journal_, sizeof(JournalMetadata), cudaMemcpyDeviceToDevice), "snapshot adapter journal");
    require_cuda(cudaMemcpy(copy.journal_positive_, journal_positive_, words_bytes, cudaMemcpyDeviceToDevice), "snapshot adapter journal positive");
    require_cuda(cudaMemcpy(copy.journal_negative_, journal_negative_, words_bytes, cudaMemcpyDeviceToDevice), "snapshot adapter journal negative");
    require_cuda(cudaMemcpy(copy.lesion_, lesion_, sizeof(LesionMetadata), cudaMemcpyDeviceToDevice), "snapshot adapter lesion");
    require_cuda(cudaMemcpy(copy.lesion_positive_, lesion_positive_, words_bytes, cudaMemcpyDeviceToDevice), "snapshot adapter lesion positive");
    require_cuda(cudaMemcpy(copy.lesion_negative_, lesion_negative_, words_bytes, cudaMemcpyDeviceToDevice), "snapshot adapter lesion negative");
  }
  void move_from(ResidentMaterial& other) noexcept {
    free_slab_ = std::exchange(other.free_slab_, nullptr);
    scalars_ = std::exchange(other.scalars_, nullptr);
    journal_ = std::exchange(other.journal_, nullptr);
    journal_positive_ = std::exchange(other.journal_positive_, nullptr);
    journal_negative_ = std::exchange(other.journal_negative_, nullptr);
    lesion_ = std::exchange(other.lesion_, nullptr);
    lesion_positive_ = std::exchange(other.lesion_positive_, nullptr);
    lesion_negative_ = std::exchange(other.lesion_negative_, nullptr);
  }

  std::uint32_t* free_slab_ = nullptr;
  MaterialScalars* scalars_ = nullptr;
  JournalMetadata* journal_ = nullptr;
  std::uint32_t* journal_positive_ = nullptr;
  std::uint32_t* journal_negative_ = nullptr;
  LesionMetadata* lesion_ = nullptr;
  std::uint32_t* lesion_positive_ = nullptr;
  std::uint32_t* lesion_negative_ = nullptr;
};

}  // namespace substrate::bcc32::resident_recurrent_material
