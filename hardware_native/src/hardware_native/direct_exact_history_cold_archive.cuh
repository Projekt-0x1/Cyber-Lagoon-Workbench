#ifndef HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_COLD_ARCHIVE_CUH
#define HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_COLD_ARCHIVE_CUH

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_network_recipe_abi.cuh"

namespace substrate::direct_adult_core {
struct DirectAdultRuntime;
}

namespace substrate::direct_network {

inline constexpr std::uint64_t kDirectExactHistoryArchiveMagic = 0x4458484152434831ull;
inline constexpr std::uint32_t kDirectExactHistoryArchiveVersion = 1u;

enum class DirectExactHistoryArchiveStatus : std::uint32_t {
  archived = 0u,
  not_quiescent,
  not_sealed,
  invalid_state,
  capacity_exhausted,
  io_error,
  content_mismatch,
};

struct DirectExactHistoryArchiveDescriptor {
  std::uint64_t magic;
  std::uint64_t object_bytes;
  recipe::Root256 subject;
  recipe::Root256 predecessor;
  std::uint64_t page_ordinal;
  std::uint64_t first_sequence;
  std::uint64_t last_sequence;
  std::uint64_t record_count;
  std::uint64_t prefix_root_before;
  std::uint64_t prefix_root_after;
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t record_bytes;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectExactHistoryArchiveDescriptor> &&
              std::is_trivial_v<DirectExactHistoryArchiveDescriptor> &&
              std::has_unique_object_representations_v<DirectExactHistoryArchiveDescriptor>);

struct DirectExactHistoryArchiveReceipt {
  DirectExactHistoryArchiveStatus status = DirectExactHistoryArchiveStatus::invalid_state;
  recipe::Root256 address{};
  std::uint64_t object_bytes = 0u;
  bool reused_durable_object = false;
  std::string object_path;
};

struct DirectExactHistoryArchivedPage {
  DirectExactHistoryArchiveDescriptor descriptor{};
  recipe::Root256 address{};
  std::vector<DirectExactHistoryRecord> records;
};

enum class DirectExactHistoryTierReadStatus : std::uint32_t {
  warm = 0u,
  cold,
  promoted,
  invalid_state,
  io_error,
  content_mismatch,
};

struct DirectExactHistoryTierReadReceipt {
  DirectExactHistoryTierReadStatus status = DirectExactHistoryTierReadStatus::invalid_state;
  recipe::Root256 address{};
  std::uint32_t access_count = 0u;
  std::uint32_t cold_fetch_count = 0u;
  std::uint64_t total_warm_hits = 0u;
  std::uint64_t total_cold_fetches = 0u;
};

std::uint64_t direct_exact_history_archive_object_bytes(const DirectExactHistoryHotPage& history);

DirectExactHistoryArchiveReceipt archive_direct_exact_history_page(
    direct_adult_core::DirectAdultRuntime* runtime, const char* directory,
    std::uint64_t archive_capacity_bytes);

DirectExactHistoryArchiveStatus read_direct_exact_history_archive_page(
    const char* directory, const recipe::Root256& expected_subject,
    const recipe::Root256& expected_address, DirectExactHistoryArchivedPage* output);

// ABI-stable cold-page read for runtime/observer consumers that need exact POD
// records without transferring std::vector ownership across CUDA/C++ TUs.
DirectExactHistoryArchiveStatus read_direct_exact_history_archive_records(
    const char* directory, const recipe::Root256& expected_subject,
    const recipe::Root256& expected_address, DirectExactHistoryRecord* records,
    std::uint32_t record_capacity, std::uint32_t record_bytes,
    std::uint32_t* record_count, recipe::Root256* predecessor);

// The resident index chooses promotion from an address-only access trace.  The
// host performs the requested byte transport and validates the content root;
// it supplies no semantic relevance or eviction winner.
DirectExactHistoryTierReadReceipt read_direct_exact_history_tiered(
    direct_adult_core::DirectAdultRuntime* runtime, const char* directory,
    const recipe::Root256& address, std::uint32_t resident_tick, bool predictive_migration,
    DirectExactHistoryArchivedPage* output);

// Dormant-recipe paging: a resident RecipeRevision with no live occurrences
// pages out to a content-addressed backing object and re-materializes
// byte-exactly on resident demand.  The slot state carries the durable
// identity while the resident body is absent, so every derivation edge,
// participation record and witness keeps resolving through identity rather
// than residency.  The host only moves whole objects; every content-root
// check happens on device inside the commit arms.
inline constexpr std::uint64_t kDormantRecipePageMagic = 0x44524d5250474531ull;
inline constexpr std::uint32_t kDormantRecipePageVersion = 1u;

enum class DormantRecipePageRequest : std::uint32_t {
  none = 0u,
  page_out = 1u,
  page_in = 2u,
};
enum class DormantRecipeWakeStatus : std::uint32_t {
  resident = 0u,
  requires_page_in,
  invalid_state,
};
enum class DormantRecipeColdStatus : std::uint32_t {
  stored = 0u,
  loaded,
  invalid_request,
  capacity_exhausted,
  io_error,
  content_mismatch,
};

struct DirectDormantRecipePageV1 {
  std::uint64_t magic;
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  ResidentRecipeCell cell;
  ResidentRecipeDerivation derivation;
  std::uint32_t version;
  std::uint32_t object_bytes;
};
struct DirectDormantRecipeSlotStateV1 {
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  recipe::Root256 requested_address;
  std::uint64_t slot_capacity_bytes;
  std::uint64_t reclaimed_slot_bytes;
  std::uint64_t transition_count;
  DormantRecipePageRequest request;
  std::uint32_t resident_present;
  std::uint32_t version;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectDormantRecipePageV1> &&
              std::is_standard_layout_v<DirectDormantRecipePageV1> &&
              std::has_unique_object_representations_v<DirectDormantRecipePageV1>);
static_assert(std::is_trivially_copyable_v<DirectDormantRecipeSlotStateV1> &&
              std::is_standard_layout_v<DirectDormantRecipeSlotStateV1>);

__host__ __device__ inline bool dormant_recipe_root_zero(const recipe::Root256& root) {
  for (std::uint32_t i = 0u; i < 8u; ++i)
    if (root.word[i] != 0u)
      return false;
  return true;
}
__host__ __device__ inline recipe::Root256 dormant_recipe_page_address(
    const DirectDormantRecipePageV1& page) {
  return recipe::content_root(&page, sizeof(page));
}
__host__ __device__ inline std::uint32_t resident_live_occurrence_count(
    const direct_adult_core::ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count, const ResidentRecipeCell& cell) {
  std::uint32_t live = 0u;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i)
    if (occurrences[i].state == direct_adult_core::kResidentRecipeOccurrenceLive &&
        occurrences[i].logical_recipe_id == cell.logical_recipe_id)
      ++live;
  return live;
}
__host__ __device__ inline bool dormant_recipe_page_valid(
    const DirectDormantRecipePageV1& page) {
  return page.magic == kDormantRecipePageMagic && page.version == kDormantRecipePageVersion &&
         page.object_bytes == sizeof(page) && page.logical_recipe_id != 0u &&
         page.revision_identity != 0u &&
         page.cell.logical_recipe_id == page.logical_recipe_id &&
         page.cell.revision_identity == page.revision_identity &&
         page.derivation.logical_recipe_id == page.logical_recipe_id &&
         page.derivation.revision_identity == page.revision_identity;
}
__host__ __device__ inline bool make_dormant_recipe_page(
    const ResidentRecipeCell& cell, const ResidentRecipeDerivation& derivation,
    DirectDormantRecipePageV1* page) {
  if (page == nullptr || cell.logical_recipe_id == 0u || cell.revision_identity == 0u ||
      cell.logical_recipe_id != derivation.logical_recipe_id ||
      cell.revision_identity != derivation.revision_identity)
    return false;
  DirectDormantRecipePageV1 candidate{};
  candidate.magic = kDormantRecipePageMagic;
  candidate.logical_recipe_id = cell.logical_recipe_id;
  candidate.revision_identity = cell.revision_identity;
  candidate.cell = cell;
  candidate.derivation = derivation;
  candidate.version = kDormantRecipePageVersion;
  candidate.object_bytes = sizeof(candidate);
  *page = candidate;
  return true;
}
__host__ __device__ inline bool initialize_dormant_recipe_slot_state(
    const ResidentRecipeCell& cell, std::uint64_t capacity_bytes,
    DirectDormantRecipeSlotStateV1* state) {
  if (state == nullptr || cell.logical_recipe_id == 0u || cell.revision_identity == 0u ||
      capacity_bytes < sizeof(DirectDormantRecipePageV1))
    return false;
  *state = {};
  state->logical_recipe_id = cell.logical_recipe_id;
  state->revision_identity = cell.revision_identity;
  state->slot_capacity_bytes = capacity_bytes;
  state->resident_present = 1u;
  state->version = kDormantRecipePageVersion;
  return true;
}
__host__ __device__ inline bool request_dormant_recipe_page_out(
    const ResidentRecipeCell& cell, const ResidentRecipeDerivation& derivation,
    const direct_adult_core::ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count, const DirectDormantRecipePageV1& page,
    DirectDormantRecipeSlotStateV1* state) {
  if (state == nullptr || !dormant_recipe_page_valid(page) ||
      state->version != kDormantRecipePageVersion || state->resident_present != 1u ||
      state->request != DormantRecipePageRequest::none ||
      cell.logical_recipe_id != state->logical_recipe_id ||
      cell.revision_identity != state->revision_identity ||
      derivation.logical_recipe_id != state->logical_recipe_id ||
      derivation.revision_identity != state->revision_identity ||
      page.logical_recipe_id != state->logical_recipe_id ||
      page.revision_identity != state->revision_identity ||
      state->slot_capacity_bytes < sizeof(page))
    return false;
  if (resident_live_occurrence_count(occurrences, occurrence_count, cell) != 0u)
    return false;
  state->requested_address = dormant_recipe_page_address(page);
  state->request = DormantRecipePageRequest::page_out;
  return !dormant_recipe_root_zero(state->requested_address);
}
__host__ __device__ inline bool commit_dormant_recipe_page_out(
    const recipe::Root256& persisted_address, ResidentRecipeCell* cell,
    ResidentRecipeDerivation* derivation, DirectDormantRecipeSlotStateV1* state) {
  if (cell == nullptr || derivation == nullptr || state == nullptr ||
      state->request != DormantRecipePageRequest::page_out ||
      dormant_recipe_root_zero(persisted_address) ||
      persisted_address != state->requested_address)
    return false;
  *cell = ResidentRecipeCell{};
  *derivation = ResidentRecipeDerivation{};
  state->reclaimed_slot_bytes = sizeof(DirectDormantRecipePageV1);
  state->resident_present = 0u;
  state->request = DormantRecipePageRequest::none;
  ++state->transition_count;
  return true;
}
__host__ __device__ inline DormantRecipeWakeStatus wake_dormant_recipe(
    DirectDormantRecipeSlotStateV1* state) {
  if (state == nullptr || state->version != kDormantRecipePageVersion ||
      state->request != DormantRecipePageRequest::none)
    return DormantRecipeWakeStatus::invalid_state;
  if (state->resident_present != 0u)
    return DormantRecipeWakeStatus::resident;
  if (dormant_recipe_root_zero(state->requested_address))
    return DormantRecipeWakeStatus::invalid_state;
  state->request = DormantRecipePageRequest::page_in;
  return DormantRecipeWakeStatus::requires_page_in;
}
__host__ __device__ inline bool commit_dormant_recipe_page_in(
    const DirectDormantRecipePageV1& staged, ResidentRecipeCell* cell,
    ResidentRecipeDerivation* derivation, DirectDormantRecipeSlotStateV1* state) {
  if (cell == nullptr || derivation == nullptr || state == nullptr ||
      state->request != DormantRecipePageRequest::page_in || state->resident_present != 0u ||
      !dormant_recipe_page_valid(staged) ||
      dormant_recipe_root_zero(state->requested_address) ||
      dormant_recipe_page_address(staged) != state->requested_address ||
      staged.logical_recipe_id != state->logical_recipe_id ||
      staged.revision_identity != state->revision_identity)
    return false;
  *cell = staged.cell;
  *derivation = staged.derivation;
  state->reclaimed_slot_bytes = 0u;
  state->resident_present = 1u;
  state->request = DormantRecipePageRequest::none;
  ++state->transition_count;
  return true;
}

inline std::string dormant_recipe_object_path(const char* directory,
                                              const recipe::Root256& address) {
  static const char hex[] = "0123456789abcdef";
  std::string name(directory);
  name += "/dormant-recipe-";
  for (std::uint32_t word : address.word)
    for (std::int32_t shift = 28; shift >= 0; shift -= 4)
      name += hex[(word >> shift) & 0xfu];
  return name + ".bin";
}

// Bulk-binary backing transport.  The host never inspects paged bytes: it
// places or fetches one whole object named by the device-produced content
// address and refuses wholesale on missing, partial or colliding objects.
inline DormantRecipeColdStatus store_dormant_recipe_page(
    const char* directory, const DirectDormantRecipeSlotStateV1& request,
    const DirectDormantRecipePageV1& page, std::string* object_path) {
  if (directory == nullptr || object_path == nullptr ||
      request.request != DormantRecipePageRequest::page_out ||
      request.resident_present != 1u || !dormant_recipe_page_valid(page) ||
      dormant_recipe_page_address(page) != request.requested_address)
    return DormantRecipeColdStatus::invalid_request;
  if (request.slot_capacity_bytes < sizeof(page))
    return DormantRecipeColdStatus::capacity_exhausted;
  if (::mkdir(directory, 0755) != 0 && errno != EEXIST)
    return DormantRecipeColdStatus::io_error;
  const std::string path = dormant_recipe_object_path(directory, request.requested_address);
  if (FILE* existing = std::fopen(path.c_str(), "rb")) {
    DirectDormantRecipePageV1 present{};
    const bool same = std::fread(&present, 1, sizeof(present), existing) == sizeof(present) &&
                      std::memcmp(&present, &page, sizeof(page)) == 0;
    std::fclose(existing);
    if (!same)
      return DormantRecipeColdStatus::content_mismatch;
    *object_path = path;
    return DormantRecipeColdStatus::stored;
  }
  const std::string temporary = path + ".tmp";
  FILE* output = std::fopen(temporary.c_str(), "wb");
  if (output == nullptr)
    return DormantRecipeColdStatus::io_error;
  const bool written =
      std::fwrite(&page, 1, sizeof(page), output) == sizeof(page) &&
      std::fflush(output) == 0;
  std::fclose(output);
  if (!written || std::rename(temporary.c_str(), path.c_str()) != 0)
    return DormantRecipeColdStatus::io_error;
  *object_path = path;
  return DormantRecipeColdStatus::stored;
}

inline DormantRecipeColdStatus load_dormant_recipe_page(
    const char* directory, const DirectDormantRecipeSlotStateV1& request,
    DirectDormantRecipePageV1* page, std::string* object_path) {
  if (directory == nullptr || page == nullptr || object_path == nullptr ||
      request.request != DormantRecipePageRequest::page_in ||
      request.resident_present != 0u || dormant_recipe_root_zero(request.requested_address))
    return DormantRecipeColdStatus::invalid_request;
  const std::string path = dormant_recipe_object_path(directory, request.requested_address);
  FILE* input = std::fopen(path.c_str(), "rb");
  if (input == nullptr)
    return DormantRecipeColdStatus::io_error;
  DirectDormantRecipePageV1 staged{};
  const bool whole = std::fread(&staged, 1, sizeof(staged), input) == sizeof(staged) &&
                     std::fgetc(input) == EOF;
  std::fclose(input);
  if (!whole)
    return DormantRecipeColdStatus::io_error;
  *page = staged;
  *object_path = path;
  return DormantRecipeColdStatus::loaded;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_COLD_ARCHIVE_CUH
