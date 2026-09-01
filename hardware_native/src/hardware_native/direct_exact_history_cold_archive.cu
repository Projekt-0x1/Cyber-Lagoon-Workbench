#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_exact_history_cold_archive.cuh"
#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_network_recipe_abi.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_network {
namespace {

enum class FileReadStatus { ok, missing, error };

using direct_adult::DirectResourceEcologyState;
using direct_adult::DirectResourcePoolKind;
using direct_adult_core::AdultExecutionAuthority;
using direct_adult_core::DirectAdultRuntime;

__global__ void reserve_archive_bytes(DirectResourceEcologyState* ecology,
                                      std::uint64_t capacity_bytes,
                                      std::uint64_t object_bytes) {
  auto* pool = direct_adult::direct_ecology_pool(
      ecology, DirectResourcePoolKind::checkpoint_future_state);
  if (pool != nullptr && pool->capacity_units == 0u && pool->charged_units == 0u &&
      pool->live_units == 0u && pool->reserved_units == 0u &&
      pool->bytes_per_unit == 0u)
    pool->capacity_units = capacity_bytes;
  if (pool != nullptr && pool->capacity_units == capacity_bytes)
    direct_adult::device_reserve_pool_units(
        ecology, DirectResourcePoolKind::checkpoint_future_state, object_bytes);
}

__device__ bool same_root(const recipe::Root256& left,
                          const recipe::Root256& right) {
  for (std::uint32_t index = 0u; index < 8u; ++index)
    if (left.word[index] != right.word[index]) return false;
  return true;
}

__global__ void acknowledge_archive_page(
    DirectExactHistoryHotPage* history, DirectResourceEcologyState* ecology,
    std::uint64_t expected_next_sequence, std::uint64_t expected_prefix_root,
    std::uint32_t expected_slots, recipe::Root256 expected_predecessor,
    recipe::Root256 address, std::uint64_t object_bytes,
    std::uint64_t archive_capacity_bytes) {
  if (history == nullptr) return;
  auto* pool = direct_adult::direct_ecology_pool(
      ecology, DirectResourcePoolKind::checkpoint_future_state);
  if (pool == nullptr || history->sealed == 0u ||
      history->next_sequence != expected_next_sequence ||
      history->prefix_root != expected_prefix_root ||
      history->committed_slots != expected_slots ||
      history->phase_kind != DirectExactHistoryKind::empty ||
      !same_root(history->archive_chain_head, expected_predecessor) ||
      pool->capacity_units != archive_capacity_bytes || pool->bytes_per_unit != 0u ||
      pool->live_units != history->archived_bytes ||
      pool->charged_units != history->archived_bytes + object_bytes ||
      pool->reserved_units != object_bytes ||
      !direct_adult::device_commit_pool_units(
          ecology, DirectResourcePoolKind::checkpoint_future_state, object_bytes))
    return;
  history->archived_record_count += history->committed_slots;
  history->archived_bytes += object_bytes;
  history->archive_capacity_bytes = archive_capacity_bytes;
  history->archive_chain_head = address;
  ++history->archived_pages;
  history->page_prefix_root = history->prefix_root;
  history->committed_slots = 0u;
  history->sealed = 0u;
  history->phase_base = 0u;
  history->phase_width = 0u;
  history->phase_kind = DirectExactHistoryKind::empty;
  history->phase_admitted = 0u;
  history->phase_tick = 0u;
  history->last_phase_records = 0u;
  for (std::uint32_t index = 0u; index < kDirectExactHistoryHotPageCapacity;
       ++index)
    history->records[index] = {};
}

__device__ std::uint32_t tier_entry_for(DirectExactHistoryTierState* tiers,
                                        const recipe::Root256& address) {
  for (std::uint32_t index = 0u; index < tiers->entry_count; ++index)
    if (tiers->entries[index].valid != 0u && same_root(tiers->entries[index].address, address))
      return index;
  std::uint32_t index = tiers->entry_count;
  if (index < kDirectExactHistoryTierIndexCapacity) {
    ++tiers->entry_count;
  } else {
    index = 0u;
    for (std::uint32_t candidate = 1u; candidate < kDirectExactHistoryTierIndexCapacity;
         ++candidate)
      if (tiers->entries[candidate].last_access_tick < tiers->entries[index].last_access_tick)
        index = candidate;
  }
  tiers->entries[index] = {};
  tiers->entries[index].address = address;
  tiers->entries[index].valid = 1u;
  return index;
}

__global__ void register_warm_archive_page(ResidentDevelopmentState* development,
                                           recipe::Root256 address) {
  if (development == nullptr)
    return;
  DirectExactHistoryTierState& tiers = development->exact_history_tiers;
  if (tiers.promotion_threshold == 0u)
    tiers.promotion_threshold = kDirectExactHistoryPromotionAccesses;
  (void)tier_entry_for(&tiers, address);
  tiers.promotion_pending = 0u;
}

__global__ void record_tier_access(ResidentDevelopmentState* development, recipe::Root256 address,
                                   std::uint32_t resident_tick, bool predictive_migration) {
  if (development == nullptr)
    return;
  DirectExactHistoryTierState& tiers = development->exact_history_tiers;
  if (tiers.promotion_threshold == 0u)
    tiers.promotion_threshold = kDirectExactHistoryPromotionAccesses;
  const std::uint32_t index = tier_entry_for(&tiers, address);
  DirectExactHistoryTierIndexEntry& entry = tiers.entries[index];
  ++entry.access_count;
  entry.last_access_tick = resident_tick;
  if (tiers.warm.valid != 0u && same_root(tiers.warm.address, address)) {
    ++tiers.warm_hits;
    tiers.promotion_pending = 0u;
    return;
  }
  ++entry.cold_fetch_count;
  ++tiers.cold_fetches;
  if (predictive_migration && entry.access_count >= tiers.promotion_threshold) {
    tiers.promotion_index = index;
    tiers.promotion_pending = 1u;
  } else {
    tiers.promotion_pending = 0u;
  }
}

__global__ void finish_tier_promotion(ResidentDevelopmentState* development,
                                      recipe::Root256 address, std::uint32_t expected_index) {
  if (development == nullptr)
    return;
  DirectExactHistoryTierState& tiers = development->exact_history_tiers;
  if (tiers.promotion_pending != 0u && tiers.promotion_index == expected_index &&
      expected_index < tiers.entry_count &&
      same_root(tiers.entries[expected_index].address, address) && tiers.warm.valid != 0u &&
      same_root(tiers.warm.address, address))
    tiers.promotion_pending = 0u;
}

bool synchronize_archive_boundary(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->brain == nullptr ||
      runtime->execution_authority != AdultExecutionAuthority::host_stepped ||
      runtime->is_persistent_running)
    return false;
  return cudaStreamSynchronize(runtime->stream) == cudaSuccess &&
         cudaStreamSynchronize(runtime->transport_stream) == cudaSuccess &&
         cudaStreamSynchronize(runtime->persistent_stream) == cudaSuccess &&
         runtime->execution_authority == AdultExecutionAuthority::host_stepped &&
         !runtime->is_persistent_running;
}

bool zero_root(const recipe::Root256& root) {
  for (std::uint32_t word : root.word)
    if (word != 0u)
      return false;
  return true;
}

std::string address_name(const recipe::Root256& address) {
  char name[8u * 8u + 5u]{};
  char* cursor = name;
  for (std::uint32_t word : address.word) {
    std::snprintf(cursor, 9u, "%08x", word);
    cursor += 8u;
  }
  std::memcpy(cursor, ".dxh", 5u);
  return name;
}

std::string object_path(const char* directory, const recipe::Root256& address) {
  std::string path(directory);
  if (!path.empty() && path.back() != '/')
    path.push_back('/');
  path += address_name(address);
  return path;
}

FileReadStatus read_file(const std::string& path, std::vector<unsigned char>* bytes) {
  const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0)
    return errno == ENOENT ? FileReadStatus::missing : FileReadStatus::error;
  struct stat metadata {};
  if (::fstat(file, &metadata) != 0 || metadata.st_size < 0 ||
      static_cast<std::uint64_t>(metadata.st_size) >
          sizeof(DirectExactHistoryArchiveDescriptor) +
              sizeof(DirectExactHistoryRecord) * kDirectExactHistoryHotPageCapacity) {
    ::close(file);
    return FileReadStatus::error;
  }
  bytes->resize(static_cast<std::size_t>(metadata.st_size));
  std::size_t offset = 0u;
  while (offset < bytes->size()) {
    const ssize_t count = ::read(file, bytes->data() + offset, bytes->size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      ::close(file);
      return FileReadStatus::error;
    }
    offset += static_cast<std::size_t>(count);
  }
  return ::close(file) == 0 ? FileReadStatus::ok : FileReadStatus::error;
}

bool write_all(int file, const std::vector<unsigned char>& bytes) {
  std::size_t offset = 0u;
  while (offset < bytes.size()) {
    const ssize_t count = ::write(file, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

class ArchiveLock {
 public:
  explicit ArchiveLock(const char* directory) {
    const std::string path = std::string(directory) + "/.direct-exact-history.lock";
    file_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (file_ >= 0 && ::flock(file_, LOCK_EX) != 0) {
      ::close(file_);
      file_ = -1;
    }
  }
  ~ArchiveLock() {
    if (file_ >= 0) {
      ::flock(file_, LOCK_UN);
      ::close(file_);
    }
  }
  bool valid() const { return file_ >= 0; }

 private:
  int file_ = -1;
};

bool sync_directory(const char* directory) {
  const int file = ::open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (file < 0) return false;
  const bool synced = ::fsync(file) == 0;
  const bool closed = ::close(file) == 0;
  return synced && closed;
}

bool sync_object(const char* directory, const std::string& path) {
  const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0) return false;
  const bool synced = ::fsync(file) == 0;
  const bool closed = ::close(file) == 0;
  return synced && closed && sync_directory(directory);
}

bool durably_remove_object(const char* directory, const std::string& path) {
  if (::unlink(path.c_str()) != 0 && errno != ENOENT) return false;
  return sync_directory(directory);
}

bool persist_object(const char* directory, const std::string& path,
                    const std::vector<unsigned char>& bytes, bool* reused,
                    bool* created) {
  *reused = false;
  *created = false;
  std::vector<unsigned char> existing;
  const FileReadStatus present = read_file(path, &existing);
  if (present == FileReadStatus::ok) {
    *reused = existing == bytes;
    return *reused && sync_object(directory, path);
  }
  if (present == FileReadStatus::error) return false;
  std::string temporary = path + ".tmp.XXXXXX";
  std::vector<char> writable(temporary.begin(), temporary.end());
  writable.push_back('\0');
  const int file = ::mkstemp(writable.data());
  if (file < 0) return false;
  temporary = writable.data();
  ::fcntl(file, F_SETFD, FD_CLOEXEC);
  const bool file_ok = write_all(file, bytes) && ::fsync(file) == 0;
  const bool close_ok = ::close(file) == 0;
  if (file_ok && close_ok && ::link(temporary.c_str(), path.c_str()) == 0) {
    *created = true;
  } else if (file_ok && close_ok && errno == EEXIST) {
    existing.clear();
    *reused = read_file(path, &existing) == FileReadStatus::ok && existing == bytes;
  }
  if (::unlink(temporary.c_str()) != 0) {
    if (*created) durably_remove_object(directory, path);
    return false;
  }
  if (!*created && !*reused) return false;
  if (*created ? !sync_directory(directory) : !sync_object(directory, path)) return false;
  existing.clear();
  return read_file(path, &existing) == FileReadStatus::ok && existing == bytes;
}

bool valid_hot_page(const DirectExactHistoryHotPage& history) {
  if (history.sealed == 0u || history.committed_slots == 0u ||
      history.committed_slots > kDirectExactHistoryHotPageCapacity ||
      history.phase_kind != DirectExactHistoryKind::empty || history.phase_admitted != 0u ||
      history.phase_width != 0u ||
      history.archived_record_count + history.committed_slots != history.next_sequence ||
      history.archived_bytes > history.archive_capacity_bytes ||
      (history.archived_pages == 0u &&
       (history.archived_record_count != 0u || history.archived_bytes != 0u ||
        history.archive_capacity_bytes != 0u || !zero_root(history.archive_chain_head) ||
        history.page_prefix_root != 0u)) ||
      (history.archived_pages != 0u &&
       (history.archived_record_count == 0u || history.archive_capacity_bytes == 0u ||
        zero_root(history.archive_chain_head))))
    return false;
  std::uint64_t root = history.page_prefix_root;
  for (std::uint32_t index = 0u; index < history.committed_slots; ++index) {
    const DirectExactHistoryRecord& record = history.records[index];
    if (record.kind == DirectExactHistoryKind::empty ||
        record.sequence != history.archived_record_count + index + 1u)
      return false;
    root = exact_history_fold_record(root, record);
  }
  return root == history.prefix_root;
}

std::vector<unsigned char> archive_bytes(const DirectExactHistoryHotPage& history,
                                         const recipe::Root256& subject) {
  DirectExactHistoryArchiveDescriptor descriptor{};
  descriptor.magic = kDirectExactHistoryArchiveMagic;
  descriptor.object_bytes = direct_exact_history_archive_object_bytes(history);
  descriptor.subject = subject;
  descriptor.predecessor = history.archive_chain_head;
  descriptor.page_ordinal = history.archived_pages + 1u;
  descriptor.first_sequence = history.archived_record_count + 1u;
  descriptor.last_sequence = history.next_sequence;
  descriptor.record_count = history.committed_slots;
  descriptor.prefix_root_before = history.page_prefix_root;
  descriptor.prefix_root_after = history.prefix_root;
  descriptor.version = kDirectExactHistoryArchiveVersion;
  descriptor.header_bytes = sizeof(descriptor);
  descriptor.record_bytes = sizeof(DirectExactHistoryRecord);
  std::vector<unsigned char> bytes(descriptor.object_bytes);
  std::memcpy(bytes.data(), &descriptor, sizeof(descriptor));
  std::memcpy(bytes.data() + sizeof(descriptor), history.records,
              sizeof(DirectExactHistoryRecord) * history.committed_slots);
  return bytes;
}

DirectExactHistoryArchiveStatus decode_archive_bytes(const std::vector<unsigned char>& bytes,
                                                     const recipe::Root256& expected_subject,
                                                     const recipe::Root256& expected_address,
                                                     DirectExactHistoryArchivedPage* output) {
  if (output == nullptr || bytes.size() < sizeof(DirectExactHistoryArchiveDescriptor) ||
      recipe::content_root(bytes.data(), bytes.size()) != expected_address)
    return DirectExactHistoryArchiveStatus::content_mismatch;
  DirectExactHistoryArchiveDescriptor descriptor{};
  std::memcpy(&descriptor, bytes.data(), sizeof(descriptor));
  if (descriptor.magic != kDirectExactHistoryArchiveMagic ||
      descriptor.version != kDirectExactHistoryArchiveVersion ||
      descriptor.header_bytes != sizeof(descriptor) ||
      descriptor.record_bytes != sizeof(DirectExactHistoryRecord) ||
      descriptor.object_bytes != bytes.size() || descriptor.subject != expected_subject ||
      descriptor.page_ordinal == 0u || descriptor.first_sequence == 0u ||
      descriptor.last_sequence < descriptor.first_sequence || descriptor.record_count == 0u ||
      descriptor.reserved != 0u || descriptor.record_count > kDirectExactHistoryHotPageCapacity ||
      (descriptor.page_ordinal == 1u && !zero_root(descriptor.predecessor)) ||
      (descriptor.page_ordinal != 1u && zero_root(descriptor.predecessor)) ||
      descriptor.object_bytes !=
          sizeof(descriptor) + sizeof(DirectExactHistoryRecord) * descriptor.record_count ||
      descriptor.last_sequence - descriptor.first_sequence + 1u != descriptor.record_count)
    return DirectExactHistoryArchiveStatus::content_mismatch;
  DirectExactHistoryArchivedPage candidate{};
  candidate.descriptor = descriptor;
  candidate.address = expected_address;
  candidate.records.resize(descriptor.record_count);
  std::memcpy(candidate.records.data(), bytes.data() + sizeof(descriptor),
              sizeof(DirectExactHistoryRecord) * descriptor.record_count);
  std::uint64_t root = descriptor.prefix_root_before;
  for (std::uint64_t index = 0u; index < descriptor.record_count; ++index) {
    const DirectExactHistoryRecord& record = candidate.records[index];
    if (record.kind == DirectExactHistoryKind::empty ||
        record.sequence != descriptor.first_sequence + index)
      return DirectExactHistoryArchiveStatus::content_mismatch;
    root = exact_history_fold_record(root, record);
  }
  if (root != descriptor.prefix_root_after)
    return DirectExactHistoryArchiveStatus::content_mismatch;
  *output = std::move(candidate);
  return DirectExactHistoryArchiveStatus::archived;
}

std::vector<unsigned char> encode_archived_page(const DirectExactHistoryArchivedPage& page) {
  const std::size_t size =
      sizeof(page.descriptor) + sizeof(DirectExactHistoryRecord) * page.records.size();
  std::vector<unsigned char> bytes(size);
  std::memcpy(bytes.data(), &page.descriptor, sizeof(page.descriptor));
  std::memcpy(bytes.data() + sizeof(page.descriptor), page.records.data(),
              sizeof(DirectExactHistoryRecord) * page.records.size());
  return bytes;
}

bool make_warm_page(const recipe::Root256& address, const std::vector<unsigned char>& bytes,
                    DirectExactHistoryWarmPage* warm) {
  if (warm == nullptr || bytes.empty() || bytes.size() > kDirectExactHistoryWarmObjectCapacity ||
      recipe::content_root(bytes.data(), bytes.size()) != address)
    return false;
  *warm = {};
  warm->address = address;
  warm->object_bytes = bytes.size();
  warm->valid = 1u;
  std::memcpy(warm->bytes, bytes.data(), bytes.size());
  return true;
}

}  // namespace

std::uint64_t direct_exact_history_archive_object_bytes(const DirectExactHistoryHotPage& history) {
  if (history.committed_slots > kDirectExactHistoryHotPageCapacity)
    return 0u;
  return sizeof(DirectExactHistoryArchiveDescriptor) +
         sizeof(DirectExactHistoryRecord) * history.committed_slots;
}

DirectExactHistoryArchiveStatus read_direct_exact_history_archive_page(
    const char* directory, const recipe::Root256& expected_subject,
    const recipe::Root256& expected_address, DirectExactHistoryArchivedPage* output) {
  if (directory == nullptr || directory[0] == '\0' || output == nullptr ||
      zero_root(expected_address))
    return DirectExactHistoryArchiveStatus::invalid_state;
  std::vector<unsigned char> bytes;
  const FileReadStatus loaded = read_file(object_path(directory, expected_address), &bytes);
  if (loaded != FileReadStatus::ok)
    return loaded == FileReadStatus::missing ? DirectExactHistoryArchiveStatus::io_error
                                             : DirectExactHistoryArchiveStatus::content_mismatch;
  return decode_archive_bytes(bytes, expected_subject, expected_address, output);
}

DirectExactHistoryArchiveStatus read_direct_exact_history_archive_records(
    const char* directory, const recipe::Root256& expected_subject,
    const recipe::Root256& expected_address, DirectExactHistoryRecord* records,
    std::uint32_t record_capacity, std::uint32_t record_bytes,
    std::uint32_t* record_count, recipe::Root256* predecessor) {
  if (records == nullptr || record_count == nullptr || predecessor == nullptr ||
      record_bytes != sizeof(DirectExactHistoryRecord))
    return DirectExactHistoryArchiveStatus::invalid_state;
  DirectExactHistoryArchivedPage page{};
  const DirectExactHistoryArchiveStatus status = read_direct_exact_history_archive_page(
      directory, expected_subject, expected_address, &page);
  if (status != DirectExactHistoryArchiveStatus::archived) return status;
  if (page.records.size() > record_capacity ||
      page.records.size() > static_cast<std::size_t>(UINT32_MAX))
    return DirectExactHistoryArchiveStatus::invalid_state;
  if (!page.records.empty())
    std::memcpy(records, page.records.data(),
                page.records.size() * sizeof(DirectExactHistoryRecord));
  *record_count = static_cast<std::uint32_t>(page.records.size());
  *predecessor = page.descriptor.predecessor;
  return DirectExactHistoryArchiveStatus::archived;
}

DirectExactHistoryTierReadReceipt read_direct_exact_history_tiered(
    DirectAdultRuntime* runtime, const char* directory, const recipe::Root256& address,
    std::uint32_t resident_tick, bool predictive_migration,
    DirectExactHistoryArchivedPage* output) {
  DirectExactHistoryTierReadReceipt receipt{};
  receipt.address = address;
  if (runtime == nullptr || runtime->brain == nullptr || runtime->brain->development == nullptr ||
      directory == nullptr || directory[0] == '\0' || output == nullptr || zero_root(address) ||
      !synchronize_archive_boundary(runtime))
    return receipt;

  record_tier_access<<<1, 1, 0, runtime->stream>>>(runtime->brain->development, address,
                                                   resident_tick, predictive_migration);
  auto tiers = std::make_unique<DirectExactHistoryTierState>();
  if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(runtime->stream) != cudaSuccess ||
      cudaMemcpy(tiers.get(), &runtime->brain->development->exact_history_tiers, sizeof(*tiers),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    receipt.status = DirectExactHistoryTierReadStatus::io_error;
    return receipt;
  }

  std::uint32_t index = kDirectExactHistoryTierIndexCapacity;
  for (std::uint32_t candidate = 0u; candidate < tiers->entry_count; ++candidate)
    if (tiers->entries[candidate].valid != 0u && tiers->entries[candidate].address == address) {
      index = candidate;
      break;
    }
  if (index == kDirectExactHistoryTierIndexCapacity)
    return receipt;
  receipt.access_count = tiers->entries[index].access_count;
  receipt.cold_fetch_count = tiers->entries[index].cold_fetch_count;
  receipt.total_warm_hits = tiers->warm_hits;
  receipt.total_cold_fetches = tiers->cold_fetches;

  DirectExactHistoryArchivedPage candidate{};
  if (tiers->warm.valid != 0u && tiers->warm.address == address) {
    if (tiers->warm.object_bytes == 0u ||
        tiers->warm.object_bytes > kDirectExactHistoryWarmObjectCapacity) {
      receipt.status = DirectExactHistoryTierReadStatus::content_mismatch;
      return receipt;
    }
    std::vector<unsigned char> bytes(
        tiers->warm.bytes, tiers->warm.bytes + static_cast<std::size_t>(tiers->warm.object_bytes));
    if (decode_archive_bytes(bytes, runtime->brain->birth_root, address, &candidate) !=
        DirectExactHistoryArchiveStatus::archived) {
      receipt.status = DirectExactHistoryTierReadStatus::content_mismatch;
      return receipt;
    }
    receipt.status = DirectExactHistoryTierReadStatus::warm;
    *output = std::move(candidate);
    return receipt;
  }

  const DirectExactHistoryArchiveStatus loaded = read_direct_exact_history_archive_page(
      directory, runtime->brain->birth_root, address, &candidate);
  if (loaded != DirectExactHistoryArchiveStatus::archived) {
    receipt.status = loaded == DirectExactHistoryArchiveStatus::content_mismatch
                         ? DirectExactHistoryTierReadStatus::content_mismatch
                         : DirectExactHistoryTierReadStatus::io_error;
    return receipt;
  }
  receipt.status = DirectExactHistoryTierReadStatus::cold;
  if (predictive_migration && tiers->promotion_pending != 0u && tiers->promotion_index == index) {
    const std::vector<unsigned char> bytes = encode_archived_page(candidate);
    auto warm = std::make_unique<DirectExactHistoryWarmPage>();
    if (!make_warm_page(address, bytes, warm.get()) ||
        cudaMemcpyAsync(&runtime->brain->development->exact_history_tiers.warm, warm.get(),
                        sizeof(*warm), cudaMemcpyHostToDevice, runtime->stream) != cudaSuccess) {
      receipt.status = DirectExactHistoryTierReadStatus::io_error;
      return receipt;
    }
    finish_tier_promotion<<<1, 1, 0, runtime->stream>>>(runtime->brain->development, address,
                                                        index);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(runtime->stream) != cudaSuccess) {
      receipt.status = DirectExactHistoryTierReadStatus::io_error;
      return receipt;
    }
    receipt.status = DirectExactHistoryTierReadStatus::promoted;
  }
  *output = std::move(candidate);
  return receipt;
}

DirectExactHistoryArchiveReceipt archive_direct_exact_history_page(
    DirectAdultRuntime* runtime, const char* directory,
    std::uint64_t archive_capacity_bytes) {
  DirectExactHistoryArchiveReceipt receipt{};
  if (runtime == nullptr || runtime->brain == nullptr ||
      runtime->brain->development == nullptr ||
      runtime->brain->resource_ecology == nullptr || directory == nullptr ||
      directory[0] == '\0' || archive_capacity_bytes == 0u)
    return receipt;
  if (!synchronize_archive_boundary(runtime)) {
    receipt.status = DirectExactHistoryArchiveStatus::not_quiescent;
    return receipt;
  }
  ArchiveLock archive_lock(directory);
  if (!archive_lock.valid()) {
    receipt.status = DirectExactHistoryArchiveStatus::io_error;
    return receipt;
  }
  DirectBrain* brain = runtime->brain;
  DirectExactHistoryHotPage history{};
  auto tiers = std::make_unique<DirectExactHistoryTierState>();
  DirectResourceEcologyState ecology{};
  if (cudaMemcpy(&history, &brain->development->exact_history, sizeof(history),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(tiers.get(), &brain->development->exact_history_tiers, sizeof(*tiers),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&ecology, brain->resource_ecology, sizeof(ecology), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
    return receipt;
  if (history.sealed == 0u) {
    receipt.status = DirectExactHistoryArchiveStatus::not_sealed;
    return receipt;
  }
  if (!valid_hot_page(history) || (history.archive_capacity_bytes != 0u &&
                                   history.archive_capacity_bytes != archive_capacity_bytes))
    return receipt;
  if (history.archived_pages != 0u) {
    DirectExactHistoryArchivedPage predecessor{};
    if (read_direct_exact_history_archive_page(directory, brain->birth_root,
                                               history.archive_chain_head, &predecessor) !=
            DirectExactHistoryArchiveStatus::archived ||
        predecessor.descriptor.page_ordinal != history.archived_pages ||
        predecessor.descriptor.last_sequence != history.archived_record_count ||
        predecessor.descriptor.prefix_root_after != history.page_prefix_root)
      return receipt;
  }
  std::vector<unsigned char> bytes = archive_bytes(history, brain->birth_root);
  receipt.object_bytes = bytes.size();
  const std::uint32_t pool_index = static_cast<std::uint32_t>(
      DirectResourcePoolKind::checkpoint_future_state);
  const auto& pool_before = ecology.pools[pool_index];
  if ((pool_before.capacity_units != 0u &&
       pool_before.capacity_units != archive_capacity_bytes) ||
      pool_before.bytes_per_unit != 0u ||
      pool_before.charged_units != history.archived_bytes ||
      pool_before.live_units != history.archived_bytes ||
      pool_before.reserved_units != 0u)
    return receipt;
  const auto rollback_resident = [&] {
    const bool history_ok = cudaMemcpy(
        &brain->development->exact_history, &history, sizeof(history),
        cudaMemcpyHostToDevice) == cudaSuccess;
    const bool tiers_ok = cudaMemcpy(&brain->development->exact_history_tiers, tiers.get(),
                                     sizeof(*tiers), cudaMemcpyHostToDevice) == cudaSuccess;
    const bool ecology_ok = cudaMemcpy(
        brain->resource_ecology, &ecology, sizeof(ecology),
        cudaMemcpyHostToDevice) == cudaSuccess;
    return history_ok && tiers_ok && ecology_ok;
  };
  reserve_archive_bytes<<<1, 1, 0, runtime->stream>>>(
      brain->resource_ecology, archive_capacity_bytes, bytes.size());
  DirectResourceEcologyState reserved{};
  if (cudaGetLastError() != cudaSuccess ||
      cudaStreamSynchronize(runtime->stream) != cudaSuccess ||
      cudaMemcpy(&reserved, brain->resource_ecology, sizeof(reserved),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    rollback_resident();
    return receipt;
  }
  const auto& reserved_pool = reserved.pools[pool_index];
  if (reserved_pool.capacity_units != archive_capacity_bytes ||
      reserved_pool.bytes_per_unit != 0u ||
      reserved_pool.live_units != history.archived_bytes ||
      reserved_pool.charged_units != history.archived_bytes + bytes.size() ||
      reserved_pool.reserved_units != bytes.size()) {
    rollback_resident();
    receipt.status = DirectExactHistoryArchiveStatus::capacity_exhausted;
    return receipt;
  }
  receipt.address = recipe::content_root(bytes.data(), bytes.size());
  receipt.object_path = object_path(directory, receipt.address);
  bool reused = false, created = false;
  if (!persist_object(directory, receipt.object_path, bytes, &reused, &created)) {
    std::vector<unsigned char> collision;
    receipt.status =
        read_file(receipt.object_path, &collision) == FileReadStatus::ok && collision != bytes
            ? DirectExactHistoryArchiveStatus::content_mismatch
            : DirectExactHistoryArchiveStatus::io_error;
    if (created) durably_remove_object(directory, receipt.object_path);
    rollback_resident();
    return receipt;
  }
  DirectExactHistoryArchivedPage verified{};
  if (read_direct_exact_history_archive_page(directory, brain->birth_root, receipt.address,
                                             &verified) !=
      DirectExactHistoryArchiveStatus::archived) {
    if (created) durably_remove_object(directory, receipt.object_path);
    rollback_resident();
    receipt.status = DirectExactHistoryArchiveStatus::content_mismatch;
    return receipt;
  }
  auto warm = std::make_unique<DirectExactHistoryWarmPage>();
  if (!make_warm_page(receipt.address, bytes, warm.get()) ||
      cudaMemcpyAsync(&brain->development->exact_history_tiers.warm, warm.get(), sizeof(*warm),
                      cudaMemcpyHostToDevice, runtime->stream) != cudaSuccess) {
    if (created)
      durably_remove_object(directory, receipt.object_path);
    rollback_resident();
    receipt.status = DirectExactHistoryArchiveStatus::invalid_state;
    return receipt;
  }
  register_warm_archive_page<<<1, 1, 0, runtime->stream>>>(brain->development, receipt.address);
  acknowledge_archive_page<<<1, 1, 0, runtime->stream>>>(
      &brain->development->exact_history, brain->resource_ecology,
      history.next_sequence, history.prefix_root, history.committed_slots,
      history.archive_chain_head, receipt.address, bytes.size(),
      archive_capacity_bytes);
  DirectExactHistoryHotPage acknowledged{};
  auto committed_tiers = std::make_unique<DirectExactHistoryTierState>();
  DirectResourceEcologyState committed{};
  if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(runtime->stream) != cudaSuccess ||
      cudaMemcpy(&acknowledged, &brain->development->exact_history, sizeof(acknowledged),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(committed_tiers.get(), &brain->development->exact_history_tiers,
                 sizeof(*committed_tiers), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&committed, brain->resource_ecology, sizeof(committed), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    if (created) durably_remove_object(directory, receipt.object_path);
    rollback_resident();
    receipt.status = DirectExactHistoryArchiveStatus::io_error;
    return receipt;
  }
  const auto& committed_pool = committed.pools[pool_index];
  if (acknowledged.archive_chain_head != receipt.address ||
      acknowledged.archived_pages != history.archived_pages + 1u ||
      acknowledged.archived_bytes != history.archived_bytes + bytes.size() ||
      committed_tiers->warm.valid == 0u || committed_tiers->warm.address != receipt.address ||
      committed_tiers->warm.object_bytes != bytes.size() ||
      committed_pool.capacity_units != archive_capacity_bytes ||
      committed_pool.charged_units != acknowledged.archived_bytes ||
      committed_pool.live_units != acknowledged.archived_bytes ||
      committed_pool.reserved_units != 0u) {
    if (created) durably_remove_object(directory, receipt.object_path);
    rollback_resident();
    receipt.status = DirectExactHistoryArchiveStatus::invalid_state;
    return receipt;
  }
  receipt.status = DirectExactHistoryArchiveStatus::archived;
  receipt.reused_durable_object = reused;
  return receipt;
}

}  // namespace substrate::direct_network
