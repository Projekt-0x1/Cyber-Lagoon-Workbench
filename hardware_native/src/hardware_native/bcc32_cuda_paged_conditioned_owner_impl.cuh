#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hardware_native/bcc32_conditioned_learning_matter.hpp"
#include "hardware_native/bcc32_cuda_paged_resident_route_bank.cuh"

namespace bcc32::paged_conditioned_owner {

namespace bank = bcc32::paged_resident_credit;
using substrate::bcc32::ConditionedMatterDeviceCredit;
using substrate::bcc32::ConditionedMatterDeviceKey;
using substrate::bcc32::SiteWord;

constexpr std::uint64_t kCheckpointMagic = UINT64_C(0x31524e574f435047);
constexpr std::uint32_t kCheckpointVersion = 1u;
constexpr std::uint32_t kPageSize = 4096u;

struct CheckpointHeader {
  std::uint64_t magic = kCheckpointMagic;
  std::uint32_t version = kCheckpointVersion;
  std::uint32_t capacity = 0u;
  std::uint32_t page_size = 0u;
  std::uint32_t page_count = 0u;
  std::uint32_t directory_capacity = 0u;
  std::uint32_t free_word_count = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t physical_hash = 0u;
  std::uint64_t matter_bits = 0u;
};

inline void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
inline void write_plain(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output) throw std::runtime_error("paged owner checkpoint write failed");
}

template <typename T>
inline T read_plain(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error("paged owner checkpoint truncated");
  return value;
}

inline std::uint32_t next_power_of_two(std::uint64_t value) {
  if (value == 0u || value > (UINT64_C(1) << 31u))
    throw std::runtime_error("paged owner directory extent overflow");
  std::uint32_t result = 1u;
  while (result < value) result <<= 1u;
  return result;
}

static __global__ void adapt_credit_kernel(
    const ConditionedMatterDeviceCredit* source, std::uint32_t count,
    bank::CreditEvent* destination) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || source == nullptr || destination == nullptr) return;
  const ConditionedMatterDeviceCredit event = source[index];
  bank::CreditEvent adapted{};
  adapted.valid = event.valid;
  adapted.polarity = event.polarity;
  adapted.key =
      bank::RouteKey{event.anchor, event.previous, event.next, 0u};
  if (event.valid != 0u && event.polarity != 1 && event.polarity != -1) {
    adapted.key = bank::RouteKey{};
    adapted.polarity = 1;
  }
  destination[index] = adapted;
}

static __global__ void publish_conductance_kernel(
    bank::PagedBankView view, const ConditionedMatterDeviceKey* keys,
    std::uint32_t count, std::uint32_t* output) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || keys == nullptr || output == nullptr) return;
  const ConditionedMatterDeviceKey key = keys[index];
  const bank::RouteKey route_key{key.anchor, key.previous, key.next, 0u};
  const std::uint32_t slot = bank::find_route(view, route_key);
  const bank::RouteMatter* route = bank::route_at(view, slot);
  if (route == nullptr || route->lesion_active != 0u) {
    output[index] = 0u;
    return;
  }
  const std::int32_t signed_value =
      static_cast<std::int32_t>(__popc(route->positive_word)) -
      static_cast<std::int32_t>(__popc(route->negative_word));
  output[index] =
      signed_value > 0 ? static_cast<std::uint32_t>(signed_value) : 0u;
}

static __global__ void transact_conditioned_kernel(
    bank::PagedBankView view, const bank::CreditEvent* events,
    std::uint32_t event_count, bank::JournalEntry* journal,
    std::uint32_t journal_capacity, bank::TransactionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  bank::TransactionReceipt local{};
  local.requested = event_count;
  *receipt = local;
  if (!bank::valid_shape(view) || events == nullptr || journal == nullptr ||
      journal_capacity < event_count) {
    local.code = bank::OperationCode::kInvalidInput;
    local.rejected = 1u;
    *receipt = local;
    return;
  }
  if (atomicCAS(&view.scalars->transaction_lock, 0u, 1u) != 0u) {
    local.code = bank::OperationCode::kBusy;
    local.rejected = 1u;
    *receipt = local;
    return;
  }
  local.before = *view.scalars;
  local.before.transaction_lock = 0u;
  if (view.scalars->state_epoch == UINT64_MAX) {
    view.scalars->transaction_lock = 0u;
    local.code = bank::OperationCode::kInvalidInput;
    local.rejected = 1u;
    local.after = *view.scalars;
    *receipt = local;
    return;
  }

  for (std::uint32_t event_index = 0u; event_index < event_count;
       ++event_index) {
    const bank::CreditEvent event = events[event_index];
    if (event.valid == 0u || event.polarity == 0) {
      ++local.abstained;
      continue;
    }
    ++local.valid;
    if (!bank::valid_key(event.key)) {
      local.code = bank::OperationCode::kRejected;
      local.rejected = 1u;
      local.failure_index = event_index;
      bank::rollback_entries(view, journal, local.journal_count, local.before);
      local.journal_count = 0u;
      local.admitted = 0u;
      local.inserted = 0u;
      local.after = *view.scalars;
      *receipt = local;
      return;
    }

    std::uint32_t bucket = bank::kInvalidIndex;
    std::uint32_t slot = bank::find_route(view, event.key, &bucket);
    bool inserted = false;
    bank::RouteMatter* route = bank::route_at(view, slot);
    if (route == nullptr && event.polarity < 0) {
      ++local.abstained;
      continue;
    }
    if (route != nullptr) {
      if (route->lesion_active != 0u) {
        local.code = bank::OperationCode::kRejected;
        local.rejected = 1u;
        local.failure_index = event_index;
        bank::rollback_entries(view, journal, local.journal_count,
                               local.before);
        local.journal_count = 0u;
        local.admitted = 0u;
        local.inserted = 0u;
        local.after = *view.scalars;
        *receipt = local;
        return;
      }
      const std::int32_t conductance =
          static_cast<std::int32_t>(__popc(route->positive_word)) -
          static_cast<std::int32_t>(__popc(route->negative_word));
      if ((event.polarity > 0 && conductance >= kConductanceCeiling) ||
          (event.polarity < 0 && conductance <= 0)) {
        ++local.abstained;
        continue;
      }
    } else {
      slot = bank::provision_route(view, event.key, &bucket, &inserted);
      route = bank::route_at(view, slot);
      if (slot == bank::kInvalidSlot || route == nullptr) {
        local.code = bank::OperationCode::kRejected;
        local.rejected = 1u;
        local.failure_index = event_index;
        bank::rollback_entries(view, journal, local.journal_count,
                               local.before);
        local.journal_count = 0u;
        local.admitted = 0u;
        local.inserted = 0u;
        local.after = *view.scalars;
        *receipt = local;
        return;
      }
    }

    bank::JournalEntry entry{};
    entry.route_before = inserted ? bank::RouteMatter{} : *route;
    entry.slot = slot;
    entry.bucket = bucket;
    entry.inserted = inserted ? 1u : 0u;
    SiteWord* destination =
        event.polarity > 0 ? &route->positive_word : &route->negative_word;
    SiteWord quantum = 0u;
    if (!bank::acquire_quantum(view, *destination, &entry.reservoir_index,
                               &quantum, &entry.reservoir_before)) {
      if (inserted) {
        *route = bank::RouteMatter{};
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
  if (local.admitted != 0u) {
    ++view.scalars->committed_transactions;
    ++view.scalars->state_epoch;
  }
  view.scalars->transaction_lock = 0u;
  local.code = bank::OperationCode::kOk;
  local.after = *view.scalars;
  *receipt = local;
}

class PagedConditionedOwnerEngine {
 public:
  PagedConditionedOwnerEngine() = default;
  explicit PagedConditionedOwnerEngine(std::uint32_t capacity) {
    reset(capacity);
  }
  ~PagedConditionedOwnerEngine() { release(); }

  PagedConditionedOwnerEngine(const PagedConditionedOwnerEngine&) = delete;
  PagedConditionedOwnerEngine& operator=(const PagedConditionedOwnerEngine&) =
      delete;

  PagedConditionedOwnerEngine(PagedConditionedOwnerEngine&& other) noexcept {
    move_from(std::move(other));
  }
  PagedConditionedOwnerEngine& operator=(
      PagedConditionedOwnerEngine&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  void reset(std::uint32_t capacity) {
    release();
    capacity_ = capacity;
    if (capacity == 0u) return;
    page_count_ = (capacity + kPageSize - 1u) / kPageSize;
    directory_capacity_ =
        next_power_of_two(static_cast<std::uint64_t>(capacity) * 2u);
    if (capacity > std::numeric_limits<std::uint32_t>::max() / 2u)
      throw std::runtime_error("paged owner reservoir extent overflow");
    free_word_count_ = capacity * 2u;

    host_pages_.assign(page_count_, nullptr);
    for (bank::RouteMatter*& page : host_pages_)
      cuda_require(cudaMalloc(&page, kPageSize * sizeof(bank::RouteMatter)),
                   "allocate paged owner route page");
    cuda_require(cudaMalloc(&device_pages_,
                            page_count_ * sizeof(bank::RouteMatter*)),
                 "allocate paged owner page table");
    cuda_require(cudaMemcpy(device_pages_, host_pages_.data(),
                            page_count_ * sizeof(bank::RouteMatter*),
                            cudaMemcpyHostToDevice),
                 "upload paged owner page table");
    cuda_require(cudaMalloc(&directory_,
                            directory_capacity_ * sizeof(std::uint32_t)),
                 "allocate paged owner directory");
    cuda_require(
        cudaMalloc(&free_reservoir_, free_word_count_ * sizeof(SiteWord)),
        "allocate paged owner reservoir");
    cuda_require(cudaMalloc(&scalars_, sizeof(bank::OwnerScalars)),
                 "allocate paged owner scalars");
    cuda_require(cudaMalloc(&measure_, sizeof(bank::PhysicalMeasure)),
                 "allocate paged owner measure");
    view_ = {device_pages_, page_count_, kPageSize, capacity_, directory_,
             directory_capacity_, free_reservoir_, free_word_count_, scalars_};
    bank::initialize_kernel<<<256u, 256u>>>(view_, 0xffffffffu);
    synchronize("initialize paged conditioned owner");
  }

  [[nodiscard]] std::uint32_t capacity() const { return capacity_; }

  [[nodiscard]] std::uint32_t size() const {
    if (capacity_ == 0u) return 0u;
    bank::OwnerScalars scalars{};
    cuda_require(cudaMemcpy(&scalars, scalars_, sizeof(scalars),
                            cudaMemcpyDeviceToHost),
                 "read paged owner size");
    return scalars.route_count;
  }

  [[nodiscard]] std::uint32_t remaining_capacity() const {
    return capacity_ - size();
  }

  [[nodiscard]] bank::PagedBankView device_view() const {
    return view_;
  }

  ConsumeReceipt consume_device_batch(
      const ConditionedMatterDeviceCredit* device_events,
      std::uint32_t count) {
    if (count == 0u) return {};
    if (capacity_ == 0u || device_events == nullptr)
      throw std::runtime_error("invalid paged owner device batch");
    ensure_scratch(count);
    adapt_credit_kernel<<<blocks_for(count), 256u>>>(
        device_events, count, events_);
    bank::PagedBankView transaction_view = view_;
    bank::CreditEvent* transaction_events = events_;
    std::uint32_t transaction_count = count;
    bank::JournalEntry* transaction_journal = journal_;
    std::uint32_t transaction_capacity = scratch_capacity_;
    bank::TransactionReceipt* transaction_receipt = receipt_;
    void* transaction_arguments[] = {
        &transaction_view, &transaction_events, &transaction_count,
        &transaction_journal, &transaction_capacity, &transaction_receipt};
    cuda_require(cudaLaunchKernel(
                     reinterpret_cast<const void*>(transact_conditioned_kernel),
                     dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, transaction_arguments, 0u,
                     nullptr),
                 "launch paged owner conditioned transaction");
    synchronize("consume paged owner device batch");
    const bank::TransactionReceipt receipt = copy_receipt();
    return {receipt.requested, receipt.valid, receipt.admitted,
            receipt.abstained, receipt.inserted, receipt.rejected,
            receipt.failure_index};
  }

  ConsumeReceipt consume_host_batch(
      const std::vector<ConditionedMatterDeviceCredit>& events) {
    if (events.empty()) return {};
    ConditionedMatterDeviceCredit* device_events = nullptr;
    cuda_require(cudaMalloc(&device_events, events.size() * sizeof(events[0])),
                 "allocate paged owner host bridge");
    try {
      cuda_require(cudaMemcpy(device_events, events.data(),
                              events.size() * sizeof(events[0]),
                              cudaMemcpyHostToDevice),
                   "upload paged owner host bridge");
      const ConsumeReceipt result = consume_device_batch(
          device_events, static_cast<std::uint32_t>(events.size()));
      cudaFree(device_events);
      return result;
    } catch (...) {
      cudaFree(device_events);
      throw;
    }
  }

  void publish_conductance_device(const ConditionedMatterDeviceKey* keys,
                                  std::uint32_t count,
                                  std::uint32_t* output) const {
    if (count == 0u) return;
    if (capacity_ == 0u || keys == nullptr || output == nullptr)
      throw std::runtime_error("invalid paged owner conductance publication");
    publish_conductance_kernel<<<blocks_for(count), 256u>>>(view_, keys, count,
                                                            output);
    synchronize("publish paged owner conductance");
  }

  bank::LesionReceipt lesion_route(bank::RouteKey key) {
    if (capacity_ == 0u) return {};
    bank::LesionReceipt* device_receipt = nullptr;
    cuda_require(cudaMalloc(&device_receipt, sizeof(*device_receipt)),
                 "allocate paged owner lesion receipt");
    try {
      bank::PagedBankView lesion_view = view_;
      bank::RouteKey lesion_key = key;
      bank::LesionReceipt* lesion_output = device_receipt;
      void* lesion_arguments[] = {&lesion_view, &lesion_key, &lesion_output};
      cuda_require(cudaLaunchKernel(
                       reinterpret_cast<const void*>(bank::lesion_kernel),
                       dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, lesion_arguments, 0u,
                       nullptr),
                   "launch paged owner route lesion");
      synchronize("lesion paged owner route");
      bank::LesionReceipt receipt{};
      cuda_require(cudaMemcpy(&receipt, device_receipt, sizeof(receipt),
                              cudaMemcpyDeviceToHost),
                   "copy paged owner lesion receipt");
      cudaFree(device_receipt);
      return receipt;
    } catch (...) {
      cudaFree(device_receipt);
      throw;
    }
  }

  bool restore_route_lesion(const bank::LesionReceipt& receipt) {
    if (capacity_ == 0u) return false;
    bank::LesionReceipt* device_receipt = nullptr;
    bank::OperationReceipt* device_result = nullptr;
    cuda_require(cudaMalloc(&device_receipt, sizeof(*device_receipt)),
                 "allocate paged owner restore receipt");
    try {
      cuda_require(cudaMalloc(&device_result, sizeof(*device_result)),
                   "allocate paged owner restore result");
      cuda_require(cudaMemcpy(device_receipt, &receipt, sizeof(receipt),
                              cudaMemcpyHostToDevice),
                   "upload paged owner restore receipt");
      bank::PagedBankView restore_view = view_;
      bank::LesionReceipt* restore_receipt = device_receipt;
      bank::OperationReceipt* restore_result = device_result;
      void* restore_arguments[] = {
          &restore_view, &restore_receipt, &restore_result};
      cuda_require(cudaLaunchKernel(
                       reinterpret_cast<const void*>(bank::restore_lesion_kernel),
                       dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, restore_arguments, 0u,
                       nullptr),
                   "launch paged owner lesion restore");
      synchronize("restore paged owner route lesion");
      bank::OperationReceipt result{};
      cuda_require(cudaMemcpy(&result, device_result, sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "copy paged owner restore result");
      cudaFree(device_result);
      cudaFree(device_receipt);
      return result.code == bank::OperationCode::kOk;
    } catch (...) {
      cudaFree(device_result);
      cudaFree(device_receipt);
      throw;
    }
  }

  [[nodiscard]] bank::PhysicalMeasure physical_measure() const {
    if (capacity_ == 0u) return {};
    bank::PagedBankView measure_view = view_;
    bank::PhysicalMeasure* measure_output = measure_;
    void* measure_arguments[] = {&measure_view, &measure_output};
    cuda_require(cudaLaunchKernel(
                     reinterpret_cast<const void*>(bank::measure_kernel),
                     dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, measure_arguments, 0u,
                     nullptr),
                 "launch paged owner physical measurement");
    synchronize("measure paged conditioned owner");
    bank::PhysicalMeasure result{};
    cuda_require(cudaMemcpy(&result, measure_, sizeof(result),
                            cudaMemcpyDeviceToHost),
                 "copy paged owner measure");
    if (result.code != bank::OperationCode::kOk)
      throw std::runtime_error("paged owner measurement failed");
    return result;
  }

  [[nodiscard]] std::uint64_t physical_hash() const {
    return physical_measure().hash;
  }

  void save(std::ostream& output) const {
    const bank::PhysicalMeasure measured = physical_measure();
    const CheckpointHeader header{kCheckpointMagic,
                                  kCheckpointVersion,
                                  capacity_,
                                  capacity_ == 0u ? 0u : kPageSize,
                                  page_count_,
                                  directory_capacity_,
                                  free_word_count_,
                                  0u,
                                  measured.hash,
                                  measured.matter_bits};
    write_plain(output, header);
    if (capacity_ == 0u) return;
    std::vector<bank::RouteMatter> page(kPageSize);
    for (const bank::RouteMatter* device_page : host_pages_) {
      cuda_require(cudaMemcpy(page.data(), device_page,
                              page.size() * sizeof(page[0]),
                              cudaMemcpyDeviceToHost),
                   "stage paged owner route page");
      output.write(reinterpret_cast<const char*>(page.data()),
                   static_cast<std::streamsize>(page.size() * sizeof(page[0])));
    }
    std::vector<std::uint32_t> directory(directory_capacity_);
    std::vector<SiteWord> reservoir(free_word_count_);
    bank::OwnerScalars scalars{};
    cuda_require(cudaMemcpy(directory.data(), directory_,
                            directory.size() * sizeof(directory[0]),
                            cudaMemcpyDeviceToHost),
                 "stage paged owner directory");
    cuda_require(cudaMemcpy(reservoir.data(), free_reservoir_,
                            reservoir.size() * sizeof(reservoir[0]),
                            cudaMemcpyDeviceToHost),
                 "stage paged owner reservoir");
    cuda_require(cudaMemcpy(&scalars, scalars_, sizeof(scalars),
                            cudaMemcpyDeviceToHost),
                 "stage paged owner scalars");
    output.write(reinterpret_cast<const char*>(directory.data()),
                 static_cast<std::streamsize>(directory.size() *
                                              sizeof(directory[0])));
    output.write(reinterpret_cast<const char*>(reservoir.data()),
                 static_cast<std::streamsize>(reservoir.size() *
                                              sizeof(reservoir[0])));
    write_plain(output, scalars);
    if (!output) throw std::runtime_error("paged owner checkpoint write failed");
  }

  static PagedConditionedOwnerEngine load(std::istream& input) {
    const CheckpointHeader header = read_plain<CheckpointHeader>(input);
    if (header.magic != kCheckpointMagic ||
        header.version != kCheckpointVersion ||
        (header.capacity != 0u &&
         (header.page_size != kPageSize ||
          header.page_count !=
              (header.capacity + kPageSize - 1u) / kPageSize ||
          header.directory_capacity !=
              next_power_of_two(static_cast<std::uint64_t>(header.capacity) *
                                2u) ||
          header.free_word_count != header.capacity * 2u)))
      throw std::runtime_error("incompatible paged owner checkpoint");
    PagedConditionedOwnerEngine result(header.capacity);
    if (header.capacity == 0u) return result;
    std::vector<bank::RouteMatter> page(kPageSize);
    for (bank::RouteMatter* device_page : result.host_pages_) {
      input.read(reinterpret_cast<char*>(page.data()),
                 static_cast<std::streamsize>(page.size() * sizeof(page[0])));
      if (!input) throw std::runtime_error("truncated paged owner route page");
      cuda_require(cudaMemcpy(device_page, page.data(),
                              page.size() * sizeof(page[0]),
                              cudaMemcpyHostToDevice),
                   "restore paged owner route page");
    }
    std::vector<std::uint32_t> directory(result.directory_capacity_);
    std::vector<SiteWord> reservoir(result.free_word_count_);
    bank::OwnerScalars scalars{};
    input.read(reinterpret_cast<char*>(directory.data()),
               static_cast<std::streamsize>(directory.size() *
                                            sizeof(directory[0])));
    input.read(reinterpret_cast<char*>(reservoir.data()),
               static_cast<std::streamsize>(reservoir.size() *
                                            sizeof(reservoir[0])));
    input.read(reinterpret_cast<char*>(&scalars), sizeof(scalars));
    if (!input || scalars.transaction_lock != 0u ||
        scalars.route_count > result.capacity_)
      throw std::runtime_error("invalid paged owner checkpoint state");
    cuda_require(cudaMemcpy(result.directory_, directory.data(),
                            directory.size() * sizeof(directory[0]),
                            cudaMemcpyHostToDevice),
                 "restore paged owner directory");
    cuda_require(cudaMemcpy(result.free_reservoir_, reservoir.data(),
                            reservoir.size() * sizeof(reservoir[0]),
                            cudaMemcpyHostToDevice),
                 "restore paged owner reservoir");
    cuda_require(cudaMemcpy(result.scalars_, &scalars, sizeof(scalars),
                            cudaMemcpyHostToDevice),
                 "restore paged owner scalars");
    const bank::PhysicalMeasure restored = result.physical_measure();
    if (restored.hash != header.physical_hash ||
        restored.matter_bits != header.matter_bits)
      throw std::runtime_error("paged owner checkpoint hash mismatch");
    return result;
  }

  static PagedConditionedOwnerEngine migrate_legacy(
      const substrate::bcc32::ConditionedLearningMatter& legacy) {
    if (legacy.capacity() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
      throw std::runtime_error("legacy conditioned capacity overflow");
    PagedConditionedOwnerEngine result(
        static_cast<std::uint32_t>(legacy.capacity()));
    const std::vector<substrate::bcc32::ConditionedMatterKey> keys =
        legacy.inventory_keys();
    const std::vector<std::uint32_t> levels = legacy.conductances(keys);
    if (keys.size() != levels.size())
      throw std::runtime_error("legacy conditioned migration shape mismatch");
    std::vector<ConditionedMatterDeviceCredit> events;
    events.reserve(4096u);
    const auto flush = [&]() {
      if (events.empty()) return;
      const ConsumeReceipt receipt = result.consume_host_batch(events);
      if (receipt.rejected != 0u || receipt.admitted == 0u)
        throw std::runtime_error("legacy conditioned migration rejected");
      events.clear();
    };
    for (std::size_t index = 0u; index < keys.size(); ++index) {
      if (levels[index] > static_cast<std::uint32_t>(kConductanceCeiling))
        throw std::runtime_error("legacy conditioned conductance overflow");
      const auto append = [&](std::int32_t polarity) {
        const auto& key = keys[index];
        events.push_back({key.anchor, key.previous, key.next, polarity,
                          static_cast<std::uint32_t>(events.size()), 1u});
        if (events.size() == events.capacity()) flush();
      };
      if (levels[index] == 0u) {
        append(1);
        append(-1);
      } else {
        for (std::uint32_t quantum = 0u; quantum < levels[index]; ++quantum)
          append(1);
      }
    }
    flush();
    if (result.size() != keys.size())
      throw std::runtime_error("legacy conditioned migration lost routes");
    return result;
  }

 private:
  static std::uint32_t blocks_for(std::uint32_t count) {
    return (count + 255u) / 256u;
  }

  static void synchronize(const char* message) {
    cuda_require(cudaGetLastError(), message);
    cuda_require(cudaDeviceSynchronize(), message);
  }

  void ensure_scratch(std::uint32_t count) {
    if (count <= scratch_capacity_) return;
    cudaFree(receipt_);
    cudaFree(journal_);
    cudaFree(events_);
    events_ = nullptr;
    journal_ = nullptr;
    receipt_ = nullptr;
    cuda_require(cudaMalloc(&events_, count * sizeof(bank::CreditEvent)),
                 "allocate paged owner events");
    cuda_require(cudaMalloc(&journal_, count * sizeof(bank::JournalEntry)),
                 "allocate paged owner journal");
    cuda_require(cudaMalloc(&receipt_, sizeof(bank::TransactionReceipt)),
                 "allocate paged owner receipt");
    scratch_capacity_ = count;
  }

  bank::TransactionReceipt copy_receipt() const {
    bank::TransactionReceipt receipt{};
    cuda_require(cudaMemcpy(&receipt, receipt_, sizeof(receipt),
                            cudaMemcpyDeviceToHost),
                 "copy paged owner transaction receipt");
    return receipt;
  }

  void release() noexcept {
    cudaFree(receipt_);
    cudaFree(journal_);
    cudaFree(events_);
    cudaFree(measure_);
    cudaFree(scalars_);
    cudaFree(free_reservoir_);
    cudaFree(directory_);
    cudaFree(device_pages_);
    for (bank::RouteMatter* page : host_pages_) cudaFree(page);
    capacity_ = 0u;
    page_count_ = 0u;
    directory_capacity_ = 0u;
    free_word_count_ = 0u;
    scratch_capacity_ = 0u;
    host_pages_.clear();
    device_pages_ = nullptr;
    directory_ = nullptr;
    free_reservoir_ = nullptr;
    scalars_ = nullptr;
    measure_ = nullptr;
    events_ = nullptr;
    journal_ = nullptr;
    receipt_ = nullptr;
    view_ = {};
  }

  void move_from(PagedConditionedOwnerEngine&& other) noexcept {
    capacity_ = std::exchange(other.capacity_, 0u);
    page_count_ = std::exchange(other.page_count_, 0u);
    directory_capacity_ = std::exchange(other.directory_capacity_, 0u);
    free_word_count_ = std::exchange(other.free_word_count_, 0u);
    scratch_capacity_ = std::exchange(other.scratch_capacity_, 0u);
    host_pages_ = std::move(other.host_pages_);
    device_pages_ = std::exchange(other.device_pages_, nullptr);
    directory_ = std::exchange(other.directory_, nullptr);
    free_reservoir_ = std::exchange(other.free_reservoir_, nullptr);
    scalars_ = std::exchange(other.scalars_, nullptr);
    measure_ = std::exchange(other.measure_, nullptr);
    events_ = std::exchange(other.events_, nullptr);
    journal_ = std::exchange(other.journal_, nullptr);
    receipt_ = std::exchange(other.receipt_, nullptr);
    view_ = other.view_;
    other.view_ = {};
  }

  std::uint32_t capacity_ = 0u;
  std::uint32_t page_count_ = 0u;
  std::uint32_t directory_capacity_ = 0u;
  std::uint32_t free_word_count_ = 0u;
  std::uint32_t scratch_capacity_ = 0u;
  std::vector<bank::RouteMatter*> host_pages_;
  bank::RouteMatter** device_pages_ = nullptr;
  std::uint32_t* directory_ = nullptr;
  SiteWord* free_reservoir_ = nullptr;
  bank::OwnerScalars* scalars_ = nullptr;
  bank::PhysicalMeasure* measure_ = nullptr;
  bank::CreditEvent* events_ = nullptr;
  bank::JournalEntry* journal_ = nullptr;
  bank::TransactionReceipt* receipt_ = nullptr;
  bank::PagedBankView view_{};
};

}  // namespace bcc32::paged_conditioned_owner
