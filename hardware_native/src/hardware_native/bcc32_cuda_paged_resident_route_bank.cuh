#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_cuda_resident_credit_bank.cuh"
#include "hardware_native/bcc32_types.cuh"

namespace bcc32::paged_resident_credit {

using substrate::bcc32::SiteWord;
using RouteKey = resident_credit::RouteKey;

constexpr std::uint32_t kInvalidSlot = 0xffffffffu;
constexpr std::uint32_t kInvalidIndex = 0xffffffffu;

enum class OperationCode : std::uint32_t {
  kNotRun = 0u,
  kOk = 1u,
  kInvalidInput = 2u,
  kBusy = 3u,
  kRejected = 4u,
  kStale = 5u,
};

struct OperationReceipt {
  OperationCode code = OperationCode::kNotRun;
};

struct RouteMatter {
  RouteKey key{};
  SiteWord positive_word = 0u;
  SiteWord negative_word = 0u;
  SiteWord lesion_positive_word = 0u;
  SiteWord lesion_negative_word = 0u;
  std::uint64_t lesion_generation = 0u;
  std::uint32_t lesion_active = 0u;
  std::uint32_t occupied = 0u;
};

struct OwnerScalars {
  std::uint32_t route_count = 0u;
  std::uint32_t free_cursor = 0u;
  std::uint32_t committed_transactions = 0u;
  std::uint64_t state_epoch = 0u;
  std::uint64_t restore_epoch = 0u;
  std::uint32_t transaction_lock = 0u;
};

// All arrays are device-owned. Route pages may be allocated independently;
// only this pointer table and the sparse directory are contiguous.
struct PagedBankView {
  RouteMatter** pages = nullptr;
  std::uint32_t page_count = 0u;
  std::uint32_t page_size = 0u;
  std::uint32_t capacity = 0u;
  std::uint32_t* directory = nullptr;
  std::uint32_t directory_capacity = 0u;
  SiteWord* free_reservoir = nullptr;
  std::uint32_t free_word_count = 0u;
  OwnerScalars* scalars = nullptr;
};

struct CreditEvent {
  RouteKey key{};
  std::int32_t polarity = 0;
  std::uint32_t valid = 0u;
};

struct JournalEntry {
  RouteMatter route_before{};
  // Factor-credit inverse uses this to reject route mutations made after commit.
  RouteMatter route_after{};
  SiteWord reservoir_before = 0u;
  std::uint32_t slot = kInvalidSlot;
  std::uint32_t bucket = kInvalidIndex;
  std::uint32_t reservoir_index = kInvalidIndex;
  std::uint32_t inserted = 0u;
};

__device__ inline void clear_journal_entries(JournalEntry* journal,
                                             std::uint32_t count) {
  for (std::uint32_t index = 0u; index < count; ++index)
    journal[index] = JournalEntry{};
}

__device__ inline bool same_route_state(const RouteMatter& left,
                                        const RouteMatter& right) {
  return resident_credit::same_key(left.key, right.key) &&
         left.positive_word == right.positive_word &&
         left.negative_word == right.negative_word &&
         left.lesion_positive_word == right.lesion_positive_word &&
         left.lesion_negative_word == right.lesion_negative_word &&
         left.lesion_generation == right.lesion_generation &&
         left.lesion_active == right.lesion_active &&
         left.occupied == right.occupied;
}

struct TransactionReceipt {
  OperationCode code = OperationCode::kNotRun;
  std::uint32_t requested = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t inserted = 0u;
  std::uint32_t journal_count = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t failure_index = kInvalidIndex;
  OwnerScalars before{};
  OwnerScalars after{};
};

struct LesionReceipt {
  OperationCode code = OperationCode::kNotRun;
  std::uint32_t valid = 0u;
  std::uint32_t slot = kInvalidSlot;
  RouteKey key{};
  SiteWord positive_word = 0u;
  SiteWord negative_word = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t restore_epoch = 0u;
};

struct PhysicalMeasure {
  OperationCode code = OperationCode::kNotRun;
  std::uint64_t hash = 0u;
  std::uint64_t content_hash = 0u;
  std::uint64_t matter_bits = 0u;
  std::uint64_t free_bits = 0u;
  std::uint64_t route_bits = 0u;
  std::uint64_t lesion_bits = 0u;
  std::uint32_t occupied_routes = 0u;
  std::uint32_t committed_transactions = 0u;
};

__host__ __device__ inline bool valid_shape(const PagedBankView& view) {
  return view.pages != nullptr && view.page_count != 0u &&
         view.page_size != 0u && view.capacity != 0u &&
         view.capacity <= view.page_count * view.page_size &&
         view.directory != nullptr && view.directory_capacity != 0u &&
         view.free_reservoir != nullptr && view.free_word_count != 0u &&
         view.scalars != nullptr;
}

__device__ inline RouteMatter* route_at(const PagedBankView& view,
                                        std::uint32_t slot) {
  if (slot >= view.capacity) return nullptr;
  const std::uint32_t page = slot / view.page_size;
  const std::uint32_t offset = slot % view.page_size;
  if (page >= view.page_count || view.pages[page] == nullptr) return nullptr;
  return &view.pages[page][offset];
}

__device__ inline std::uint64_t mix_hash(std::uint64_t hash,
                                         std::uint64_t value) {
  hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6u) +
          (hash >> 2u);
  return hash * UINT64_C(1099511628211);
}

__host__ __device__ inline bool same_owner_state(const OwnerScalars& left,
                                                 const OwnerScalars& right) {
  return left.route_count == right.route_count &&
         left.free_cursor == right.free_cursor &&
         left.committed_transactions == right.committed_transactions &&
         left.state_epoch == right.state_epoch &&
         left.restore_epoch == right.restore_epoch;
}

__device__ inline bool valid_key(const RouteKey& key) {
  return key.anchor != 0u && key.previous != 0u && key.next != 0u;
}

__device__ inline std::uint32_t directory_home(const PagedBankView& view,
                                                const RouteKey& key) {
  return static_cast<std::uint32_t>(
      resident_credit::key_hash(key) % view.directory_capacity);
}

__device__ inline std::uint32_t find_route(
    const PagedBankView& view, const RouteKey& key,
    std::uint32_t* found_bucket = nullptr) {
  if (!valid_shape(view)) return kInvalidSlot;
  const std::uint32_t home = directory_home(view, key);
  for (std::uint32_t probe = 0u; probe < view.directory_capacity; ++probe) {
    const std::uint32_t bucket =
        (home + probe) % view.directory_capacity;
    const std::uint32_t encoded = view.directory[bucket];
    if (encoded == 0u) return kInvalidSlot;
    const std::uint32_t slot = encoded - 1u;
    const RouteMatter* route = route_at(view, slot);
    if (route != nullptr && route->occupied != 0u &&
        resident_credit::same_key(route->key, key)) {
      if (found_bucket != nullptr) *found_bucket = bucket;
      return slot;
    }
  }
  return kInvalidSlot;
}

__device__ inline std::uint32_t provision_route(
    PagedBankView view, const RouteKey& key, std::uint32_t* bucket_out,
    bool* inserted_out) {
  if (bucket_out != nullptr) *bucket_out = kInvalidIndex;
  if (inserted_out != nullptr) *inserted_out = false;
  std::uint32_t existing_bucket = kInvalidIndex;
  const std::uint32_t existing =
      find_route(view, key, &existing_bucket);
  if (existing != kInvalidSlot) {
    if (bucket_out != nullptr) *bucket_out = existing_bucket;
    return existing;
  }
  if (view.scalars->route_count >= view.capacity) return kInvalidSlot;
  const std::uint32_t home = directory_home(view, key);
  for (std::uint32_t probe = 0u; probe < view.directory_capacity; ++probe) {
    const std::uint32_t bucket =
        (home + probe) % view.directory_capacity;
    if (view.directory[bucket] != 0u) continue;
    const std::uint32_t slot = view.scalars->route_count++;
    RouteMatter* route = route_at(view, slot);
    if (route == nullptr) {
      --view.scalars->route_count;
      return kInvalidSlot;
    }
    *route = RouteMatter{};
    route->key = key;
    route->occupied = 1u;
    __threadfence();
    view.directory[bucket] = slot + 1u;
    if (bucket_out != nullptr) *bucket_out = bucket;
    if (inserted_out != nullptr) *inserted_out = true;
    return slot;
  }
  return kInvalidSlot;
}

__device__ inline bool acquire_quantum(PagedBankView view,
                                       SiteWord destination,
                                       std::uint32_t* reservoir_index,
                                       SiteWord* bit,
                                       SiteWord* reservoir_before) {
  if (reservoir_index != nullptr) *reservoir_index = kInvalidIndex;
  if (bit != nullptr) *bit = 0u;
  if (reservoir_before != nullptr) *reservoir_before = 0u;
  if (__popc(destination) >= 32) return false;
  const std::uint32_t start =
      view.scalars->free_cursor % view.free_word_count;
  for (std::uint32_t offset = 0u; offset < view.free_word_count; ++offset) {
    const std::uint32_t index =
        (start + offset) % view.free_word_count;
    const SiteWord available =
        view.free_reservoir[index] & ~destination;
    if (available == 0u) continue;
    const SiteWord selected =
        static_cast<SiteWord>(1u << (__ffs(available) - 1));
    if (reservoir_index != nullptr) *reservoir_index = index;
    if (bit != nullptr) *bit = selected;
    if (reservoir_before != nullptr)
      *reservoir_before = view.free_reservoir[index];
    view.free_reservoir[index] &= ~selected;
    view.scalars->free_cursor = (index + 1u) % view.free_word_count;
    return true;
  }
  return false;
}

__device__ inline void rollback_entries(
    PagedBankView view, JournalEntry* journal, std::uint32_t count,
    const OwnerScalars& before) {
  for (std::uint32_t offset = 0u; offset < count; ++offset) {
    const std::uint32_t index = count - 1u - offset;
    const JournalEntry entry = journal[index];
    RouteMatter* route = route_at(view, entry.slot);
    if (entry.reservoir_index != kInvalidIndex)
      view.free_reservoir[entry.reservoir_index] =
          entry.reservoir_before;
    if (route != nullptr) *route = entry.route_before;
    if (entry.inserted != 0u && entry.bucket < view.directory_capacity)
      view.directory[entry.bucket] = 0u;
  }
  *view.scalars = before;
}

static __global__ void initialize_kernel(PagedBankView view,
                                         SiteWord reservoir_word) {
  if (!valid_shape(view)) return;
  for (std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
       slot < view.capacity; slot += blockDim.x * gridDim.x) {
    RouteMatter* route = route_at(view, slot);
    if (route != nullptr) *route = RouteMatter{};
  }
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < view.directory_capacity; index += blockDim.x * gridDim.x)
    view.directory[index] = 0u;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < view.free_word_count; index += blockDim.x * gridDim.x)
    view.free_reservoir[index] = reservoir_word;
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    *view.scalars = OwnerScalars{};
}

// A transaction is serialized inside one device kernel under exclusive,
// externally ordered owner access. No host decision exists between validation,
// sparse provisioning, conserved matter transfer, and commit. Invalid input or
// capacity failure restores every touched physical state before return.
static __global__ void transact_kernel(
    PagedBankView view, const CreditEvent* events, std::uint32_t event_count,
    JournalEntry* journal, std::uint32_t journal_capacity,
    TransactionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  TransactionReceipt local{};
  local.requested = event_count;
  *receipt = local;
  if (!valid_shape(view) || events == nullptr || journal == nullptr) {
    local.code = OperationCode::kInvalidInput;
    local.rejected = 1u;
    *receipt = local;
    return;
  }
  if (atomicCAS(&view.scalars->transaction_lock, 0u, 1u) != 0u) {
    local.code = OperationCode::kBusy;
    local.rejected = 1u;
    local.failure_index = 0u;
    local.before = *view.scalars;
    local.after = *view.scalars;
    *receipt = local;
    return;
  }
  local.before = *view.scalars;
  local.before.transaction_lock = 0u;
  if (journal_capacity < event_count ||
      view.scalars->state_epoch == UINT64_MAX) {
    local.code = OperationCode::kInvalidInput;
    local.rejected = 1u;
    local.failure_index = 0u;
    view.scalars->transaction_lock = 0u;
    local.after = *view.scalars;
    *receipt = local;
    return;
  }

  for (std::uint32_t event_index = 0u; event_index < event_count;
       ++event_index) {
    const CreditEvent event = events[event_index];
    if (event.valid == 0u || event.polarity == 0) {
      ++local.abstained;
      continue;
    }
    ++local.valid;
    if (!valid_key(event.key)) {
      local.code = OperationCode::kRejected;
      local.rejected = 1u;
      local.failure_index = event_index;
      rollback_entries(view, journal, local.journal_count, local.before);
      clear_journal_entries(journal, local.journal_count);
      local.journal_count = 0u;
      local.admitted = 0u;
      local.inserted = 0u;
      local.after = *view.scalars;
      *receipt = local;
      return;
    }

    std::uint32_t bucket = kInvalidIndex;
    bool inserted = false;
    const std::uint32_t slot =
        provision_route(view, event.key, &bucket, &inserted);
    if (slot == kInvalidSlot) {
      local.code = OperationCode::kRejected;
      local.rejected = 1u;
      local.failure_index = event_index;
      rollback_entries(view, journal, local.journal_count, local.before);
      clear_journal_entries(journal, local.journal_count);
      local.journal_count = 0u;
      local.admitted = 0u;
      local.inserted = 0u;
      local.after = *view.scalars;
      *receipt = local;
      return;
    }

    RouteMatter* route = route_at(view, slot);
    if (route == nullptr || route->lesion_active != 0u) {
      local.code = OperationCode::kRejected;
      local.rejected = 1u;
      local.failure_index = event_index;
      rollback_entries(view, journal, local.journal_count, local.before);
      clear_journal_entries(journal, local.journal_count);
      local.journal_count = 0u;
      local.admitted = 0u;
      local.inserted = 0u;
      local.after = *view.scalars;
      *receipt = local;
      return;
    }
    JournalEntry entry{};
    entry.route_before = inserted ? RouteMatter{} : *route;
    entry.slot = slot;
    entry.bucket = bucket;
    entry.inserted = inserted ? 1u : 0u;
    SiteWord* destination =
        event.polarity > 0 ? &route->positive_word
                           : &route->negative_word;
    SiteWord quantum = 0u;
    if (!acquire_quantum(view, *destination, &entry.reservoir_index,
                         &quantum, &entry.reservoir_before)) {
      if (inserted) {
        *route = RouteMatter{};
        view.directory[bucket] = 0u;
        --view.scalars->route_count;
      }
      ++local.abstained;
      continue;
    }
    *destination |= quantum;
    journal[local.journal_count++] = entry;
    ++local.admitted;
    local.inserted += inserted ? 1u : 0u;
  }
  ++view.scalars->committed_transactions;
  ++view.scalars->state_epoch;
  view.scalars->transaction_lock = 0u;
  local.code = OperationCode::kOk;
  local.after = *view.scalars;
  *receipt = local;
}

static __global__ void inverse_last_transaction_kernel(
    PagedBankView view, JournalEntry* journal,
    const TransactionReceipt* receipt, OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = OperationReceipt{};
  if (!valid_shape(view) || journal == nullptr || receipt == nullptr) {
    result->code = OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&view.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = OperationCode::kBusy;
    return;
  }
  OwnerScalars current = *view.scalars;
  current.transaction_lock = 0u;
  if (receipt->code != OperationCode::kOk || receipt->rejected != 0u ||
      !same_owner_state(current, receipt->after)) {
    view.scalars->transaction_lock = 0u;
    result->code = OperationCode::kStale;
    return;
  }
  rollback_entries(view, journal, receipt->journal_count, receipt->before);
  clear_journal_entries(journal, receipt->journal_count);
  result->code = OperationCode::kOk;
}

static __global__ void capture_checkpoint_kernel(PagedBankView source,
                                                 PagedBankView destination,
                                                 OperationReceipt* result) {
  if (blockIdx.x != 0u || result == nullptr) return;
  __shared__ std::uint64_t destination_restore_epoch;
  if (threadIdx.x == 0u) {
    *result = OperationReceipt{};
    if (!valid_shape(source) || !valid_shape(destination) ||
        source.capacity != destination.capacity ||
        source.directory_capacity != destination.directory_capacity ||
        source.free_word_count != destination.free_word_count ||
        destination.scalars->restore_epoch == UINT64_MAX) {
      result->code = OperationCode::kInvalidInput;
    } else {
      destination_restore_epoch = destination.scalars->restore_epoch;
      result->code = OperationCode::kOk;
    }
  }
  __syncthreads();
  if (result->code != OperationCode::kOk) return;
  for (std::uint32_t slot = threadIdx.x; slot < source.capacity;
       slot += blockDim.x) {
    RouteMatter* source_route = route_at(source, slot);
    RouteMatter* destination_route = route_at(destination, slot);
    if (source_route != nullptr && destination_route != nullptr)
      *destination_route = *source_route;
  }
  for (std::uint32_t index = threadIdx.x;
       index < source.directory_capacity; index += blockDim.x)
    destination.directory[index] = source.directory[index];
  for (std::uint32_t index = threadIdx.x; index < source.free_word_count;
       index += blockDim.x)
    destination.free_reservoir[index] = source.free_reservoir[index];
  __syncthreads();
  if (threadIdx.x == 0u) {
    *destination.scalars = *source.scalars;
    destination.scalars->restore_epoch = destination_restore_epoch + 1u;
  }
}

static __global__ void read_conductance_kernel(
    PagedBankView view, const RouteKey* keys, std::uint32_t count,
    std::int32_t* output) {
  const std::uint32_t index =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || keys == nullptr || output == nullptr) return;
  const std::uint32_t slot = find_route(view, keys[index]);
  const RouteMatter* route = route_at(view, slot);
  output[index] =
      route == nullptr
          ? 0
          : static_cast<std::int32_t>(__popc(route->positive_word)) -
                static_cast<std::int32_t>(__popc(route->negative_word));
}

static __global__ void lesion_kernel(PagedBankView view, RouteKey key,
                                     LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  *receipt = LesionReceipt{};
  if (!valid_shape(view)) {
    receipt->code = OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&view.scalars->transaction_lock, 0u, 1u) != 0u) {
    receipt->code = OperationCode::kBusy;
    return;
  }
  const std::uint32_t slot = find_route(view, key);
  RouteMatter* route = route_at(view, slot);
  if (route == nullptr || route->lesion_active != 0u ||
      route->lesion_generation == UINT64_MAX ||
      view.scalars->state_epoch == UINT64_MAX) {
    view.scalars->transaction_lock = 0u;
    receipt->code = OperationCode::kRejected;
    return;
  }
  receipt->code = OperationCode::kOk;
  receipt->valid = 1u;
  receipt->slot = slot;
  receipt->key = route->key;
  receipt->positive_word = route->positive_word;
  receipt->negative_word = route->negative_word;
  receipt->generation = route->lesion_generation + 1u;
  receipt->restore_epoch = view.scalars->restore_epoch;
  route->lesion_positive_word = route->positive_word;
  route->lesion_negative_word = route->negative_word;
  route->lesion_generation = receipt->generation;
  route->lesion_active = 1u;
  route->positive_word = 0u;
  route->negative_word = 0u;
  ++view.scalars->state_epoch;
  view.scalars->transaction_lock = 0u;
}

static __global__ void restore_lesion_kernel(PagedBankView view,
                                             const LesionReceipt* receipt,
                                             OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = OperationReceipt{};
  if (!valid_shape(view) || receipt == nullptr || receipt->valid == 0u ||
      receipt->code != OperationCode::kOk) {
    result->code = OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&view.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = OperationCode::kBusy;
    return;
  }
  RouteMatter* route = route_at(view, receipt->slot);
  if (route == nullptr || route->occupied == 0u ||
      !resident_credit::same_key(route->key, receipt->key) ||
      route->lesion_active == 0u ||
      route->lesion_generation != receipt->generation ||
      view.scalars->restore_epoch != receipt->restore_epoch ||
      view.scalars->state_epoch == UINT64_MAX ||
      route->lesion_positive_word != receipt->positive_word ||
      route->lesion_negative_word != receipt->negative_word ||
      route->positive_word != 0u || route->negative_word != 0u) {
    view.scalars->transaction_lock = 0u;
    result->code = OperationCode::kStale;
    return;
  }
  route->positive_word = receipt->positive_word;
  route->negative_word = receipt->negative_word;
  route->lesion_positive_word = 0u;
  route->lesion_negative_word = 0u;
  route->lesion_active = 0u;
  ++view.scalars->state_epoch;
  view.scalars->transaction_lock = 0u;
  result->code = OperationCode::kOk;
}

static __global__ void measure_kernel(PagedBankView view,
                                      PhysicalMeasure* measure) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || measure == nullptr) return;
  PhysicalMeasure local{};
  *measure = local;
  if (!valid_shape(view)) {
    local.code = OperationCode::kInvalidInput;
    *measure = local;
    return;
  }
  local.code = OperationCode::kOk;
  std::uint64_t hash = UINT64_C(1469598103934665603);
  std::uint64_t content_hash = UINT64_C(1469598103934665603);
  for (std::uint32_t slot = 0u; slot < view.capacity; ++slot) {
    const RouteMatter* route = route_at(view, slot);
    if (route == nullptr) continue;
    hash = mix_hash(hash, route->key.anchor);
    hash = mix_hash(hash, route->key.previous);
    hash = mix_hash(hash, route->key.next);
    hash = mix_hash(hash, route->key.region);
    hash = mix_hash(hash, route->positive_word);
    hash = mix_hash(hash, route->negative_word);
    hash = mix_hash(hash, route->lesion_positive_word);
    hash = mix_hash(hash, route->lesion_negative_word);
    hash = mix_hash(hash, route->lesion_generation);
    hash = mix_hash(hash, route->lesion_active);
    hash = mix_hash(hash, route->occupied);
    content_hash = mix_hash(content_hash, route->key.anchor);
    content_hash = mix_hash(content_hash, route->key.previous);
    content_hash = mix_hash(content_hash, route->key.next);
    content_hash = mix_hash(content_hash, route->key.region);
    content_hash = mix_hash(content_hash, route->positive_word);
    content_hash = mix_hash(content_hash, route->negative_word);
    content_hash = mix_hash(content_hash, route->lesion_positive_word);
    content_hash = mix_hash(content_hash, route->lesion_negative_word);
    content_hash = mix_hash(content_hash, route->lesion_active);
    content_hash = mix_hash(content_hash, route->occupied);
    local.route_bits += __popc(route->positive_word) +
                        __popc(route->negative_word);
    local.lesion_bits += __popc(route->lesion_positive_word) +
                         __popc(route->lesion_negative_word);
    local.occupied_routes += route->occupied != 0u ? 1u : 0u;
  }
  for (std::uint32_t index = 0u; index < view.directory_capacity; ++index) {
    hash = mix_hash(hash, view.directory[index]);
    content_hash = mix_hash(content_hash, view.directory[index]);
  }
  for (std::uint32_t index = 0u; index < view.free_word_count; ++index) {
    hash = mix_hash(hash, view.free_reservoir[index]);
    content_hash = mix_hash(content_hash, view.free_reservoir[index]);
    local.free_bits += __popc(view.free_reservoir[index]);
  }
  hash = mix_hash(hash, view.scalars->route_count);
  hash = mix_hash(hash, view.scalars->free_cursor);
  hash = mix_hash(hash, view.scalars->committed_transactions);
  hash = mix_hash(hash, view.scalars->state_epoch);
  hash = mix_hash(hash, view.scalars->restore_epoch);
  hash = mix_hash(hash, view.scalars->transaction_lock);
  content_hash = mix_hash(content_hash, view.scalars->route_count);
  content_hash = mix_hash(content_hash, view.scalars->free_cursor);
  content_hash =
      mix_hash(content_hash, view.scalars->committed_transactions);
  local.matter_bits =
      local.free_bits + local.route_bits + local.lesion_bits;
  local.committed_transactions =
      view.scalars->committed_transactions;
  local.hash = hash;
  local.content_hash = content_hash;
  *measure = local;
}

}  // namespace bcc32::paged_resident_credit
