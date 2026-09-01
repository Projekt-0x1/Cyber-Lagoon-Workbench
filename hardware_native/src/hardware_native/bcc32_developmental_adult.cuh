#pragma once

#include "bcc32_active_support.cuh"
#include "bcc32_membrane.hpp"
#include "bcc32_aperture_geometry.cuh"
#include "bcc32_cuda_executor.cuh"
#include "bcc32_developmental_population_seed.cuh"
#include "bcc32_device_ordinary_f_timeline.cuh"
#include "bcc32_grown_cloud_factor.cuh"
#include "bcc32_grown_instance_basin.cuh"
#include "bcc32_grown_form_credit_factor.cuh"
#include "bcc32_grown_sensorimotor_factor.cuh"
#include "bcc32_grown_selective_state_space.cuh"
#include "bcc32_grown_sparse_event_memory.cuh"
#include "bcc32_law.cuh"
#include "bcc32_law_identity.hpp"
#include "bcc32_raw_byte_tape.cuh"
#include "bcc32_resident_edge_bank.cuh"
#include "bcc32_resident_readout_f_route.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace substrate::bcc32::developmental_adult {

// These are the exact section-21 developmental-recursion recipe and schedule.
// They are mechanism identifiers, not language content or semantic state.
inline constexpr DevelopmentalHash kFounderHash = kFirstPopulationHash;
// Alias onto the single shared definition in bcc32_aperture_geometry.cuh --
// not a second literal. That header is also included (without a cycle) by
// every grown_*_factor.cuh fixed_physical_slot() that needs this value.
inline constexpr std::int64_t kApertureEdgeChunks =
    substrate::bcc32::kApertureEdgeChunks;
inline constexpr int kAutonomousGrowthTicks = 45;
inline constexpr int kResidencyWindowTicks = 20;
inline constexpr int kCanonicalDevelopmentTicks =
    kAutonomousGrowthTicks + kResidencyWindowTicks;
inline constexpr std::uint32_t kGermSiteCount =
    static_cast<std::uint32_t>(kPopulationSeedSiteCount);
// Overridable ONLY to falsify the aperture growth law's offset term; default
// unchanged and every shipped target uses it. The law measured at 920f90367d is
// ticks = half_width - 25 - offset, and the offset was DERIVED as the germ's
// initial +x extent. Two readings of that extent differ by one tick, so varying
// this constant discriminates them rather than fitting either.
#ifndef BCC32_FOUNDER_SPACING
#define BCC32_FOUNDER_SPACING 8
#endif
inline constexpr std::int64_t kFounderSpacing = BCC32_FOUNDER_SPACING;
inline constexpr std::array<char, 8> kCheckpointMagic{
    'B', 'C', 'C', '3', '2', 'G', 'A', '1'};
inline constexpr std::uint32_t kCheckpointVersion = 7u;
inline constexpr std::uint32_t kBoundaryWordCount = 16u;
inline constexpr std::uint32_t kRawSensoryZeroPort = 0u;
inline constexpr std::uint32_t kRawSensoryOnePort = 3u;
inline constexpr std::uint32_t kRawMotorZeroPort = 1u;
inline constexpr std::uint32_t kRawMotorOnePort = 4u;
inline constexpr std::uint32_t kMotorIdleCarrierPattern = 0x55u;

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct HostChunkMap {
  std::vector<DeviceChunkSlot> host;
  DeviceChunkSlot* device = nullptr;

  HostChunkMap() = default;
  HostChunkMap(const HostChunkMap&) = delete;
  HostChunkMap& operator=(const HostChunkMap&) = delete;
  HostChunkMap(HostChunkMap&& other) noexcept
      : host(std::move(other.host)), device(std::exchange(other.device, nullptr)) {}
  HostChunkMap& operator=(HostChunkMap&& other) noexcept {
    if (this == &other) return *this;
    if (device != nullptr) (void)cudaFree(device);
    host = std::move(other.host);
    device = std::exchange(other.device, nullptr);
    return *this;
  }
  ~HostChunkMap() {
    if (device != nullptr) (void)cudaFree(device);
  }

  [[nodiscard]] DeviceChunkMap view() const {
    return {device, static_cast<std::uint32_t>(host.size())};
  }
};

struct DeviceBoundaryWords {
  SiteWord* device = nullptr;

  DeviceBoundaryWords() = default;
  DeviceBoundaryWords(const DeviceBoundaryWords&) = delete;
  DeviceBoundaryWords& operator=(const DeviceBoundaryWords&) = delete;
  ~DeviceBoundaryWords() {
    if (device != nullptr) (void)cudaFree(device);
  }
};

inline std::int32_t chunk_index(std::span<const DeviceChunkSlot> slots,
                                std::int64_t x, std::int64_t y,
                                std::int64_t z) {
  for (std::size_t index = 0; index < slots.size(); ++index) {
    if (slots[index].chunk_x == x && slots[index].chunk_y == y &&
        slots[index].chunk_z == z) {
      return static_cast<std::int32_t>(index);
    }
  }
  return DeviceChunkSlot::kMissing;
}

inline HostChunkMap make_cube(std::int64_t edge_chunks) {
  if (edge_chunks <= 0) {
    throw std::invalid_argument("grown-adult aperture edge must be positive");
  }
  HostChunkMap map;
  for (std::int64_t x = 0; x < edge_chunks; ++x) {
    for (std::int64_t y = 0; y < edge_chunks; ++y) {
      for (std::int64_t z = 0; z < edge_chunks; ++z) {
        DeviceChunkSlot slot{};
        slot.chunk_x = x;
        slot.chunk_y = y;
        slot.chunk_z = z;
        map.host.push_back(slot);
      }
    }
  }
  for (DeviceChunkSlot& slot : map.host) {
    for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
      const Int3 offset =
          direction_offset(static_cast<Direction>(direction));
      slot.bcc_neighbors[direction] =
          chunk_index(map.host, slot.chunk_x + offset.x,
                      slot.chunk_y + offset.y, slot.chunk_z + offset.z);
    }
  }
  require_cuda(
      cudaMalloc(&map.device, map.host.size() * sizeof(DeviceChunkSlot)),
      "allocate grown-adult chunk map");
  require_cuda(cudaMemcpy(map.device, map.host.data(),
                          map.host.size() * sizeof(DeviceChunkSlot),
                          cudaMemcpyHostToDevice),
               "upload grown-adult chunk map");
  return map;
}

inline std::uint64_t slot_at(const HostChunkMap& map, std::int64_t gx,
                             std::int64_t gy, std::int64_t gz) {
  const std::int64_t edge = static_cast<std::int64_t>(kChunkEdge);
  if (gx < 0 || gy < 0 || gz < 0) {
    throw std::out_of_range("grown-adult coordinate is outside aperture");
  }
  const std::int32_t chunk =
      chunk_index(map.host, gx / edge, gy / edge, gz / edge);
  if (chunk == DeviceChunkSlot::kMissing) {
    throw std::out_of_range("grown-adult coordinate is outside aperture");
  }
  const std::uint64_t local =
      (static_cast<std::uint64_t>(gx % edge) * kChunkEdge +
       static_cast<std::uint64_t>(gy % edge)) *
          kChunkEdge +
      static_cast<std::uint64_t>(gz % edge);
  return static_cast<std::uint64_t>(chunk) * kChunkSites + local;
}

struct StateEntry {
  std::uint64_t slot = 0u;
  SiteWord word = kQ;

  friend bool operator==(const StateEntry&, const StateEntry&) = default;
};

struct GenesisManifestEntry {
  Int3 relative{};
  SiteWord word = kQ;

  friend bool operator==(const GenesisManifestEntry&,
                         const GenesisManifestEntry&) = default;
};

enum class GenesisRecipeKind : std::uint32_t {
  developmental_hash = 0u,
  explicit_manifest = 1u,
};

struct Snapshot {
  std::vector<StateEntry> entries;
  std::uint64_t completed_ticks = 0u;

  friend bool operator==(const Snapshot&, const Snapshot&) = default;
};

struct Residency {
  std::vector<std::uint64_t> slots;
  std::uint32_t occupied = 0u;
  std::uint32_t saturated = 0u;
};

struct BoundaryCounters {
  std::uint64_t sensory_bytes = 0u;
  std::uint64_t motor_bytes = 0u;
  std::uint64_t semantic_host_writes = 0u;

  friend bool operator==(const BoundaryCounters&,
                         const BoundaryCounters&) = default;
};

struct BoundarySnapshot {
  std::array<SiteWord, kBoundaryWordCount> words{};

  friend bool operator==(const BoundarySnapshot&,
                         const BoundarySnapshot&) = default;
};

struct CheckpointHeader {
  std::array<char, 8> magic{};
  std::uint32_t version = 0u;
  std::uint32_t header_bytes = 0u;
  DevelopmentalHash founder_hash{};
  std::uint32_t germ_site_count = 0u;
  std::int64_t aperture_edge_chunks = 0;
  std::uint64_t completed_ticks = 0u;
  std::uint64_t site_count = 0u;
  std::uint64_t entry_count = 0u;
  std::uint64_t state_hash = 0u;
  std::uint64_t boundary_hash = 0u;
  std::uint64_t topology_hash = 0u;
  BoundaryCounters boundary{};
  std::array<SiteWord, kBoundaryWordCount> boundary_words{};
  std::uint64_t resident_history_count = 0u;
  std::uint64_t resident_history_hash = 0u;
  std::uint64_t founder_entry_count = 0u;
  std::uint64_t founder_state_hash = 0u;
  ContentAddress law_identity{};
  ContentAddress genesis_manifest_identity{};
  std::uint64_t genesis_manifest_entry_count = 0u;
  GenesisRecipeKind genesis_recipe_kind = GenesisRecipeKind::developmental_hash;
  std::uint32_t reserved = 0u;
};

struct CheckpointEntry {
  std::uint64_t slot = 0u;
  SiteWord word = kQ;
  std::uint32_t reserved = 0u;
};

static_assert(std::is_trivially_copyable_v<CheckpointHeader>);
static_assert(std::is_trivially_copyable_v<CheckpointEntry>);

inline std::uint64_t hash_bytes(std::uint64_t hash, const void* data,
                                std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

inline std::uint64_t state_hash(const Snapshot& snapshot) {
  std::uint64_t hash = 1469598103934665603ull;
  hash = hash_bytes(hash, &snapshot.completed_ticks,
                    sizeof(snapshot.completed_ticks));
  for (const StateEntry& entry : snapshot.entries) {
    hash = hash_bytes(hash, &entry.slot, sizeof(entry.slot));
    hash = hash_bytes(hash, &entry.word, sizeof(entry.word));
  }
  return hash;
}

inline void append_manifest_u64(std::vector<std::byte>* bytes,
                                std::uint64_t value) {
  for (std::uint32_t shift = 0u; shift < 64u; shift += 8u)
    bytes->push_back(static_cast<std::byte>(value >> shift));
}

inline void append_manifest_u32(std::vector<std::byte>* bytes,
                                std::uint32_t value) {
  for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
    bytes->push_back(static_cast<std::byte>(value >> shift));
}

inline ContentAddress genesis_manifest_address(
    std::string_view recipe_id,
    std::span<const GenesisManifestEntry> ordered_entries,
    std::span<const DeviceChunkSlot> topology) {
  std::vector<std::byte> bytes;
  constexpr std::array<std::byte, 8u> tag{
      std::byte{'B'}, std::byte{'C'}, std::byte{'C'}, std::byte{'G'},
      std::byte{'E'}, std::byte{'N'}, std::byte{'5'}, std::byte{0}};
  bytes.insert(bytes.end(), tag.begin(), tag.end());
  const ContentAddress law = canonical_law_identity();
  for (const std::uint8_t byte : law.digest)
    bytes.push_back(static_cast<std::byte>(byte));
  append_manifest_u64(&bytes, law.byte_count);
  append_manifest_u64(&bytes, kChunkEdge);
  append_manifest_u64(&bytes, topology.size());
  for (const DeviceChunkSlot& chunk : topology) {
    append_manifest_u64(
        &bytes, static_cast<std::uint64_t>(chunk.chunk_x));
    append_manifest_u64(
        &bytes, static_cast<std::uint64_t>(chunk.chunk_y));
    append_manifest_u64(
        &bytes, static_cast<std::uint64_t>(chunk.chunk_z));
    for (const std::int32_t neighbor : chunk.bcc_neighbors)
      append_manifest_u32(
          &bytes, static_cast<std::uint32_t>(neighbor));
  }
  append_manifest_u64(&bytes, recipe_id.size());
  for (const char byte : recipe_id)
    bytes.push_back(static_cast<std::byte>(
        static_cast<std::uint8_t>(byte)));
  append_manifest_u64(&bytes, ordered_entries.size());
  for (const GenesisManifestEntry& entry : ordered_entries) {
    append_manifest_u64(&bytes, static_cast<std::uint64_t>(
                                    static_cast<std::int64_t>(entry.relative.x)));
    append_manifest_u64(&bytes, static_cast<std::uint64_t>(
                                    static_cast<std::int64_t>(entry.relative.y)));
    append_manifest_u64(&bytes, static_cast<std::uint64_t>(
                                    static_cast<std::int64_t>(entry.relative.z)));
    append_manifest_u32(&bytes, entry.word);
  }
  return content_address(bytes);
}

inline std::uint64_t bound_state_hash(
    const Snapshot& snapshot, const Snapshot& founder_snapshot,
    const DevelopmentalHash& founder_hash,
    const ContentAddress& law,
    const ContentAddress& manifest, GenesisRecipeKind kind,
    std::uint64_t manifest_entry_count) {
  std::uint64_t hash = state_hash(snapshot);
  hash = hash_bytes(hash, &founder_hash, sizeof(founder_hash));
  const std::uint64_t founder_state_hash = state_hash(founder_snapshot);
  hash = hash_bytes(hash, &founder_state_hash, sizeof(founder_state_hash));
  hash = hash_bytes(hash, law.digest.data(), law.digest.size());
  hash = hash_bytes(hash, &law.byte_count, sizeof(law.byte_count));
  hash = hash_bytes(hash, manifest.digest.data(), manifest.digest.size());
  hash = hash_bytes(hash, &manifest.byte_count, sizeof(manifest.byte_count));
  hash = hash_bytes(hash, &kind, sizeof(kind));
  return hash_bytes(hash, &manifest_entry_count,
                    sizeof(manifest_entry_count));
}

inline std::uint64_t resident_history_hash(
    std::span<const std::uint32_t> history) {
  return hash_bytes(1469598103934665603ull, history.data(),
                    history.size_bytes());
}

inline std::uint64_t boundary_hash(const BoundarySnapshot& snapshot) {
  return hash_bytes(1469598103934665603ull, snapshot.words.data(),
                    snapshot.words.size() * sizeof(SiteWord));
}

inline std::uint64_t topology_hash(std::span<const DeviceChunkSlot> topology) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const DeviceChunkSlot& slot : topology) {
    hash = hash_bytes(hash, &slot.chunk_x, sizeof(slot.chunk_x));
    hash = hash_bytes(hash, &slot.chunk_y, sizeof(slot.chunk_y));
    hash = hash_bytes(hash, &slot.chunk_z, sizeof(slot.chunk_z));
    hash = hash_bytes(hash, slot.bcc_neighbors, sizeof(slot.bcc_neighbors));
  }
  return hash;
}

enum class FounderVariant : std::uint8_t {
  intact,
  target_equal_mass_lesion,
  remote_equal_mass_lesion,
  far_equal_mass_lesion,
};

inline SiteWord equal_mass_lesion_word(SiteWord word) {
  return std::rotl(word, 1);
}

inline std::int64_t seed_distance_squared(const DevelopmentalSeedSite& left,
                                          const DevelopmentalSeedSite& right) {
  const std::int64_t dx = static_cast<std::int64_t>(left.x) - right.x;
  const std::int64_t dy = static_cast<std::int64_t>(left.y) - right.y;
  const std::int64_t dz = static_cast<std::int64_t>(left.z) - right.z;
  return dx * dx + dy * dy + dz * dz;
}

inline bool seed_coordinate_less(const DevelopmentalSeedSite& left,
                                  const DevelopmentalSeedSite& right) {
  if (left.x != right.x) return left.x < right.x;
  if (left.y != right.y) return left.y < right.y;
  return left.z < right.z;
}

inline std::size_t observer_lesion_seed_index(
    std::span<const DevelopmentalSeedSite> seed, FounderVariant variant) {
  if (seed.empty() || variant == FounderVariant::intact) return seed.size();

  DevelopmentalSeedSite origin{0, 0, 0, kQ};
  std::size_t target = 0u;
  for (std::size_t index = 1u; index < seed.size(); ++index) {
    const std::int64_t candidate_distance =
        seed_distance_squared(seed[index], origin);
    const std::int64_t selected_distance =
        seed_distance_squared(seed[target], origin);
    if (candidate_distance < selected_distance ||
        (candidate_distance == selected_distance &&
         seed_coordinate_less(seed[index], seed[target])))
      target = index;
  }
  if (variant == FounderVariant::target_equal_mass_lesion) return target;

  std::size_t selected = target;
  if (variant == FounderVariant::remote_equal_mass_lesion) {
    for (std::size_t index = 0u; index < seed.size(); ++index) {
      const std::int64_t candidate_distance =
          seed_distance_squared(seed[index], seed[target]);
      const std::int64_t selected_distance =
          seed_distance_squared(seed[selected], seed[target]);
      if (candidate_distance > selected_distance ||
          (candidate_distance == selected_distance &&
           seed_coordinate_less(seed[index], seed[selected])))
        selected = index;
    }
    return selected;
  }

  selected = target;
  for (std::size_t index = 0u; index < seed.size(); ++index) {
    const std::int64_t candidate_distance =
        seed_distance_squared(seed[index], origin);
    const std::int64_t selected_distance =
        seed_distance_squared(seed[selected], origin);
    if (candidate_distance > selected_distance ||
        (candidate_distance == selected_distance &&
         seed_coordinate_less(seed[index], seed[selected])))
      selected = index;
  }
  return selected;
}

inline std::vector<GenesisManifestEntry> developmental_hash_manifest(
    DevelopmentalHash hash, FounderVariant variant = FounderVariant::intact) {
  const auto seed = developmental_population_seed(hash);
  const std::size_t lesion_index = observer_lesion_seed_index(seed, variant);
  std::vector<GenesisManifestEntry> manifest;
  manifest.reserve(seed.size());
  for (std::size_t index = 0u; index < seed.size(); ++index) {
    SiteWord word = seed[index].word;
    const bool lesion = index == lesion_index;
    if (lesion) word = equal_mass_lesion_word(word);
    manifest.push_back({{seed[index].x, seed[index].y, seed[index].z}, word});
  }
  return manifest;
}

inline Int3 boundary_port_relative(std::uint32_t index) {
  if (index >= kBoundaryWordCount) {
    throw std::out_of_range("grown-adult boundary index invalid");
  }
  constexpr std::array<Int3, kBoundaryWordCount> kPorts{{
      {0, 0, 0},
      {0, 1, 0},
      {0, -1, 0},
      {0, 0, 1},
      {0, 0, -1},
      {static_cast<std::int32_t>(kFounderSpacing), 0, 0},
      {static_cast<std::int32_t>(kFounderSpacing), 1, 0},
      {static_cast<std::int32_t>(kFounderSpacing), -1, 0},
      {static_cast<std::int32_t>(kFounderSpacing), 0, 1},
      {static_cast<std::int32_t>(kFounderSpacing), 0, -1},
      {static_cast<std::int32_t>(2 * kFounderSpacing), 0, 0},
      {static_cast<std::int32_t>(2 * kFounderSpacing), 1, 0},
      {static_cast<std::int32_t>(2 * kFounderSpacing), -1, 0},
      {static_cast<std::int32_t>(2 * kFounderSpacing), 0, 1},
      {static_cast<std::int32_t>(2 * kFounderSpacing), 0, -1},
      {static_cast<std::int32_t>(2 * kFounderSpacing + 1), 0, 0},
  }};
  return kPorts[index];
}

static __global__ void exchange_boundary_word_kernel(
    SiteWord* world_words, std::uint64_t slot, SiteWord* boundary_words,
    std::uint32_t boundary_index) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const SiteWord world_word = world_words[slot];
  world_words[slot] = boundary_words[boundary_index];
  boundary_words[boundary_index] = world_word;
}

static __global__ void exchange_boundary_raw_byte_kernel(
    SiteWord* world_words, std::uint64_t zero_slot, std::uint64_t one_slot,
    SiteWord* boundary_words, std::uint32_t zero_boundary_index,
    std::uint32_t one_boundary_index) {
  if (threadIdx.x == 0u) {
    reciprocal_field_exchange<kCarrierShift, kFaceShift, 8u>(
        world_words[zero_slot], boundary_words[zero_boundary_index]);
  } else {
    reciprocal_field_exchange<kCarrierShift, kFaceShift, 8u>(
        world_words[one_slot], boundary_words[one_boundary_index]);
  }
}

// Exact event-local boundary exchange only. This is a reversible sensory
// contact, not prediction, learning, or semantic byte interpretation.
static __global__ void exchange_sensory_contact_kernel(
    SiteWord* world_words, std::uint64_t zero_slot, std::uint64_t one_slot,
    RawByteRails* device_escrow) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  RawByteRails world{world_words[zero_slot], world_words[one_slot]};
  reciprocal_raw_byte_exchange(world, *device_escrow);
  world_words[zero_slot] = world.zero;
  world_words[one_slot] = world.one;
}

// ⭐ BATCHED RECIPROCAL CONTACT -- a class-2 boundary transaction, moved into
// production because it is mechanism and was living in a test.
//
// The single-slot `exchange_sensory_contact_kernel` above handles one rail pair.
// A contact PROGRAM exchanges many world/tape bit pairs in one stage, and that
// kernel was defined inside
// bcc32_cuda_grown_adult_prediction_comparator_contract.cu, which meant a test
// held a batched write route into organism memory and the adult's world pointer
// had to be public for it to run. §13 and the anti-orbit hook both say mechanism
// belongs in a production header; the ownership work made it load-bearing.
struct ContactExchange {
  std::uint64_t slot = 0u;
  SiteWord world_bit = 0u;
  std::uint32_t tape_index = 0u;
  SiteWord tape_bit = 0u;
};

static __global__ void apply_contact_stage_kernel(SiteWord* world_words,
                                                  SiteWord* tape,
                                                  const ContactExchange* exchanges,
                                                  std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const ContactExchange exchange = exchanges[index];
  SiteWord* world_word = world_words + exchange.slot;
  SiteWord* tape_word = tape + exchange.tape_index;
  const bool world_set =
      (atomicAdd(reinterpret_cast<unsigned int*>(world_word), 0u) &
       exchange.world_bit) != 0u;
  const bool tape_set =
      (atomicAdd(reinterpret_cast<unsigned int*>(tape_word), 0u) &
       exchange.tape_bit) != 0u;
  if (world_set != tape_set) {
    atomicXor(reinterpret_cast<unsigned int*>(world_word), exchange.world_bit);
    atomicXor(reinterpret_cast<unsigned int*>(tape_word), exchange.tape_bit);
  }
}

// ⭐ THE CENSUS THAT MAKES A BATCHED RECEIPT POSSIBLE.
//
// A batch cannot report `before`/`after` as one word, so it reports the
// POPULATION of the declared bit set on each side. Three accumulators, one pass:
//
//   world -- set bits among the declared world bits. This is the organism-side
//            matter the receipt calls before/after.
//   pair  -- world + tape together. ⭐ THIS is the conserved quantity, and it is
//            summed over BOTH OWNERS deliberately. Entries 170 and 171 each
//            reported conservation from one side of a two-owner exchange and
//            were wrong both times; the swap below leaves NEITHER side
//            individually invariant by construction.
//   ident -- an order-independent FINGERPRINT of the declared descriptor list.
//   alias -- declared pairs naming the same world word AND the same world bit.
//
// 🔴 `ident` WAS DOCUMENTED AS NAMING WHICH BATCH RAN, AND IT DID NOT. The fold
// was a plain sum of a mix that is LINEAR in the slot, so any two batches with
// equal slot sums collide -- systematically, not rarely. Measured: descriptor
// slots {0,9} and {1,8} both fold to 9895606726579. The per-descriptor avalanche
// below destroys that degeneracy, but a sum is never injective, so the field is
// named a fingerprint and claims only that a changed declaration is detectable.
//
// ⛔ AND `pair` IS NOW ENTAILED, WHICH REPLACES AN EARLIER OVERCLAIM. The
// exchange kernel swaps only when the two sides disagree, so the ONLY way a
// batch can lose matter is two declared pairs naming the same world word and
// bit -- their two atomicXors cancel. That is now REFUSED before the launch
// instead of executed and reported, so on any batch that actually runs,
// conservation is a consequence of `aliased == 0` rather than an independent
// check. `alias` is the arm that can fail.
static __global__ void contact_stage_census_kernel(const SiteWord* world_words,
                                                   const SiteWord* tape,
                                                   const ContactExchange* exchanges,
                                                   std::uint32_t count,
                                                   unsigned int* world_population,
                                                   unsigned int* pair_population,
                                                   unsigned long long* fingerprint,
                                                   unsigned int* aliased) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const ContactExchange exchange = exchanges[index];
  const unsigned int world_set =
      (world_words[exchange.slot] & exchange.world_bit) != 0u ? 1u : 0u;
  const unsigned int tape_set =
      (tape[exchange.tape_index] & exchange.tape_bit) != 0u ? 1u : 0u;
  atomicAdd(world_population, world_set);
  atomicAdd(pair_population, world_set + tape_set);
  unsigned long long mixed =
      (static_cast<unsigned long long>(exchange.slot) * 1099511628211ull) ^
      (static_cast<unsigned long long>(exchange.world_bit) * 1000003ull) ^
      (static_cast<unsigned long long>(exchange.tape_index) * 2654435761ull) ^
      (static_cast<unsigned long long>(exchange.tape_bit) * 40503ull);
  // splitmix64 finalizer: makes the summand a nonlinear function of the whole
  // descriptor, so equal slot sums no longer force equal fingerprints.
  mixed ^= mixed >> 30;
  mixed *= 0xbf58476d1ce4e5b9ull;
  mixed ^= mixed >> 27;
  mixed *= 0x94d049bb133111ebull;
  mixed ^= mixed >> 31;
  atomicAdd(fingerprint, mixed);
  // Alias detection. O(n^2) over the declared list, which is bounded by the
  // stage size (single digits to low hundreds), and it runs on the same pass
  // that already reads every descriptor.
  for (std::uint32_t other = 0u; other < index; ++other) {
    const ContactExchange earlier = exchanges[other];
    if (earlier.slot == exchange.slot &&
        (earlier.world_bit & exchange.world_bit) != 0u) {
      atomicAdd(aliased, 1u);
    }
  }
}

static __global__ void scatter_state_entries_kernel(
    SiteWord* world_words, const StateEntry* entries, std::uint64_t count) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    world_words[entries[index].slot] = entries[index].word;
  }
}

// kMotor remains the learned authority. The raw motor aperture is only its
// reversible device-side surface projection.
static __device__ inline void project_sensorimotor_motor_surface(
    SiteWord* words, std::uint64_t motor_slot, std::uint64_t raw_zero_slot,
    std::uint64_t raw_one_slot) {
  const SiteWord motor = words[motor_slot];
  RawByteRails surface{words[raw_zero_slot], words[raw_one_slot]};
  if (__popc(motor) == 1 && (motor & ~0xffu) == 0u) {
    surface = with_raw_byte_carriers(surface, static_cast<std::uint8_t>(motor));
  } else {
    surface.zero = with_carriers(surface.zero, kMotorIdleCarrierPattern);
    surface.one = with_carriers(surface.one, kMotorIdleCarrierPattern);
  }
  words[raw_zero_slot] = surface.zero;
  words[raw_one_slot] = surface.one;
}

static __global__ void combined_cloud_sensorimotor_step_kernel(
    SiteWord* words, const grown_cloud_factor::DeviceLayout* cloud_layout,
    grown_cloud_factor::ContactReceipt* cloud_receipt,
    const grown_sensorimotor_factor::DeviceLayout* sensorimotor_layout,
    grown_sensorimotor_factor::DeviceInputs* sensorimotor_inputs,
    grown_sensorimotor_factor::PredictionReceipt* prediction,
    grown_sensorimotor_factor::ConsequenceReceipt* consequence,
    grown_sensorimotor_factor::TransformReceipt* transform,
    std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (!grown_cloud_factor::journal_available(words, *cloud_layout) ||
      !grown_sensorimotor_factor::journal_available(
          words, *sensorimotor_layout)) {
    *advanced = 0u;
    return;
  }
  const bool cloud_advanced =
      grown_cloud_factor::step_device(words, *cloud_layout, cloud_receipt);
  const bool sensorimotor_advanced =
      grown_sensorimotor_factor::step_device(
          words, *sensorimotor_layout, sensorimotor_inputs, prediction,
          consequence, transform);
  *advanced = cloud_advanced && sensorimotor_advanced ? 1u : 0u;
}

static __global__ void combined_cloud_sensorimotor_inverse_kernel(
    SiteWord* words, const grown_cloud_factor::DeviceLayout* cloud_layout,
    const grown_sensorimotor_factor::DeviceLayout* sensorimotor_layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  grown_sensorimotor_factor::inverse_step_device(words, *sensorimotor_layout);
  grown_cloud_factor::inverse_step_device(words, *cloud_layout);
}

inline constexpr std::uint32_t kResidentStepSucceeded = 1u << 0u;
inline constexpr std::uint32_t kResidentStepAdvancedInstance = 1u << 1u;
inline constexpr std::uint32_t kResidentStepJournaledFormCredit = 1u << 2u;
inline constexpr std::uint32_t kResidentStepAgedFormCredit = 1u << 3u;
inline constexpr std::uint32_t kResidentStepSelectiveState = 1u << 4u;
inline constexpr std::uint32_t kResidentStepSparseEventMemory = 1u << 5u;

// One coincidence-triggered association between the instance-basin owner
// factor and the cloud destination factor. Recorded per resident-factor tick
// alongside resident_step_history_ (same push-forward/pop-reverse lifecycle,
// see develop()/reverse() below) because the inverse cannot recompute which
// two physical slots were read, nor which edge-bank slot (if any) was
// written, from later state alone -- it must be told, exactly as
// form_credit's journaled/aged flags are told via the advancement bitmask.
//
// NEITHER endpoint rail is ever written (owner_slot/dest_slot below are read
// ONLY, to capture the two tokens); the relation itself lives in the
// separate bcc32_resident_edge_bank.cuh ancilla, whose writes are self-
// inverse XORs, so only "did an edge form, and at which index" needs to be
// carried -- the token values themselves double as the exact XOR delta the
// inverse needs.
struct AssociationOutcome {
  std::uint32_t fired = 0u;
  std::uint64_t owner_slot = 0u;
  std::uint64_t dest_slot = 0u;
  // Captured at the moment of coincidence detection (after this tick's own
  // factor writes), not before the whole tick -- the owner rail is written
  // by instance_basin's own step earlier in this same dispatch, so "at
  // detection" and "before the tick" differ. These are read-only captures:
  // this factor never writes owner_slot or dest_slot.
  SiteWord owner_value_before = 0u;
  SiteWord dest_value_before = 0u;
  // resident_edge_bank::EdgeAction (kActionNone when the association itself
  // did not fire, or when it fired but the edge bank was not configured).
  std::uint32_t edge_action = 0u;
  std::uint32_t edge_index = 0xffffffffu;

  // Resident RECALL: fires when exactly one basin is selected/changed AND no
  // coincidence formed this same tick (the owner is active alone). Reads the
  // edge bank on-device for an occupied record whose owner_token matches (no
  // host index ever chooses the destination), and -- if found -- applies a
  // destination-SPECIFIC physical consequence to the matching cloud cell: a
  // bit-quantum XOR-moved from that cell's OWN kCellFreeMatter field into its
  // OWN kCellSupport field. XOR against a value/complement pair is self-
  // inverse (same proof as bcc32_resident_edge_bank.cuh's file comment), so
  // recall_cell/recall_quantum below are all the inverse needs to undo it
  // exactly. Recall runs LAST in forward dispatch (after formation), so its
  // inverse must run FIRST in combined_resident_factors_inverse_kernel --
  // LIFO with respect to formation's own edge-bank undo.
  std::uint32_t recall_attempted = 0u;
  std::uint32_t recall_abstained = 0u;
  std::uint32_t recall_edge_index = 0xffffffffu;
  SiteWord recall_owner_token = 0u;
  SiteWord recall_dest_token = 0u;
  std::uint32_t recall_cell = 0xffffffffu;
  SiteWord recall_quantum = 0u;
};

static __global__ void combined_resident_factors_step_kernel(
    SiteWord* words, bool cloud_enabled,
    const grown_cloud_factor::DeviceLayout* cloud_layout,
    grown_cloud_factor::ContactReceipt* cloud_receipt,
    bool instance_enabled,
    const grown_instance_basin_factor::DeviceLayout* instance_layout,
    grown_instance_basin_factor::DeviceInputs* instance_inputs,
    grown_instance_basin_factor::StepReceipt* instance_receipt,
    bool sensorimotor_enabled,
    const grown_sensorimotor_factor::DeviceLayout* sensorimotor_layout,
    std::uint64_t raw_motor_zero_slot, std::uint64_t raw_motor_one_slot,
    grown_sensorimotor_factor::DeviceInputs* sensorimotor_inputs,
    grown_sensorimotor_factor::PredictionReceipt* sensorimotor_prediction,
    grown_sensorimotor_factor::ConsequenceReceipt* sensorimotor_consequence,
    grown_sensorimotor_factor::TransformReceipt* sensorimotor_transform,
    bool readout_enabled,
    const resident_readout_f_route::DeviceLayout* readout_layout,
    resident_readout_f_route::DeviceInputs* readout_inputs,
    resident_readout_f_route::CreditReceipt* readout_receipt,
    bool form_credit_enabled,
    const grown_form_credit_factor::DeviceLayout* form_credit_layout,
    grown_form_credit_factor::DeviceInputs* form_credit_inputs,
    grown_form_credit_factor::Receipt* form_credit_receipt,
    bool selective_state_enabled,
    const grown_selective_state_space::DeviceInputs* selective_state_inputs,
    bool sparse_event_memory_enabled,
    const grown_sparse_event_memory::DeviceInputs* sparse_event_memory_inputs,
    bool edge_bank_enabled,
    const resident_edge_bank::DeviceLayout* edge_bank_layout,
    std::uint32_t* advanced, AssociationOutcome* association_outcome) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (selective_state_enabled &&
      !grown_selective_state_space::step_available(words,
                                                   selective_state_inputs)) {
    *advanced = 0u;
    return;
  }
  if (sparse_event_memory_enabled &&
      !grown_sparse_event_memory::step_available(
          words, sparse_event_memory_inputs)) {
    *advanced = 0u;
    return;
  }
  if (cloud_enabled &&
      !grown_cloud_factor::journal_available(words, *cloud_layout)) {
    *advanced = 0u;
    return;
  }
  const bool instance_advanced =
      instance_enabled && instance_inputs != nullptr &&
      instance_inputs->staged != 0u;
  const bool form_credit_journaled =
      form_credit_enabled &&
      grown_form_credit_factor::step_requires_journal(
          words, *form_credit_layout, form_credit_inputs);
  if (instance_advanced &&
      !grown_instance_basin_factor::journal_available(words, *instance_layout)) {
    *advanced = 0u;
    return;
  }
  if (sensorimotor_enabled &&
      !grown_sensorimotor_factor::journal_available(
          words, *sensorimotor_layout)) {
    *advanced = 0u;
    return;
  }
  if (readout_enabled &&
      !resident_readout_f_route::journal_available(words, *readout_layout)) {
    *advanced = 0u;
    return;
  }
  if (form_credit_journaled &&
      !grown_form_credit_factor::journal_available(words, *form_credit_layout)) {
    *advanced = 0u;
    return;
  }
  bool ok = true;
  if (cloud_enabled) {
    ok = grown_cloud_factor::step_device(words, *cloud_layout,
                                         cloud_receipt) &&
         ok;
  }
  if (instance_enabled) {
    (void)grown_instance_basin_factor::step_device(
        words, *instance_layout, instance_inputs, instance_receipt);
  }
  if (sensorimotor_enabled) {
    ok = grown_sensorimotor_factor::step_device(
             words, *sensorimotor_layout, sensorimotor_inputs,
             sensorimotor_prediction, sensorimotor_consequence,
             sensorimotor_transform) &&
         ok;
    // Form credit owns the shared raw-motor aperture when present. The
    // sensorimotor factor remains upstream and must not overwrite that surface.
    if (!form_credit_enabled) {
      project_sensorimotor_motor_surface(
          words,
          sensorimotor_layout->rails[grown_sensorimotor_factor::value_index(
              grown_sensorimotor_factor::kMotor)],
          raw_motor_zero_slot, raw_motor_one_slot);
    }
  }
  if (readout_enabled) {
    ok = resident_readout_f_route::step_device(
             words, *readout_layout, readout_inputs, readout_receipt) &&
         ok;
  }
  std::uint32_t form_credit_step_mode = grown_form_credit_factor::kStepNone;
  if (form_credit_enabled) {
    ok = grown_form_credit_factor::step_device(
             words, *form_credit_layout, form_credit_inputs,
             form_credit_receipt, &form_credit_step_mode) &&
         ok;
  }
  // Coincidence-triggered association, LAST in forward dispatch (see
  // combined_resident_factors_inverse_kernel below, which therefore applies
  // it FIRST -- b7bee3f55c: an inverse must unwind LIFO). Owner side: exactly
  // one instance-basin basin selected or changed this step (matched XOR
  // recruited, never both for the same basin in one step). Destination
  // side: exactly one cloud cell matched this contact. Either side absent or
  // ambiguous means no fire -- unambiguous or not at all.
  AssociationOutcome association_local{};
  if (instance_enabled && cloud_enabled && instance_receipt != nullptr &&
      cloud_receipt != nullptr) {
    const SiteWord owner_candidates =
        (instance_receipt->matched_mask | instance_receipt->recruited_mask) &
        grown_instance_basin_factor::kFullMask;
    const SiteWord dest_candidates = cloud_receipt->active_mask;
    if (__popc(owner_candidates) == 1 && __popc(dest_candidates) == 1) {
      const std::uint32_t basin = static_cast<std::uint32_t>(
          __ffs(static_cast<int>(owner_candidates)) - 1);
      const std::uint32_t cell = static_cast<std::uint32_t>(
          __ffs(static_cast<int>(dest_candidates)) - 1);
      const std::uint64_t owner_slot = instance_layout->rails
          [grown_instance_basin_factor::resident_index(
              grown_instance_basin_factor::basin_field(
                  basin, grown_instance_basin_factor::kAppearance))];
      const std::uint64_t dest_slot = grown_cloud_factor::fixed_physical_slot(
          grown_cloud_factor::cell_rail(cell, grown_cloud_factor::kCellCue,
                                        0u));
      const SiteWord owner_token = words[owner_slot];
      const SiteWord dest_token = words[dest_slot];
      association_local.owner_value_before = owner_token;
      association_local.dest_value_before = dest_token;
      association_local.fired = 1u;
      association_local.owner_slot = owner_slot;
      association_local.dest_slot = dest_slot;
      // Endpoints are never written here -- only read (owner_token,
      // dest_token above). The relation is recorded in the separate
      // edge-bank ancilla instead of exchanging the two endpoints' matter.
      if (edge_bank_enabled && edge_bank_layout != nullptr) {
        const resident_edge_bank::EdgeOutcome edge_outcome =
            resident_edge_bank::form_edge_device(words, *edge_bank_layout,
                                                 owner_token, dest_token);
        association_local.edge_action = edge_outcome.action;
        association_local.edge_index = edge_outcome.edge_index;
      }
    }
  }
  // Resident RECALL -- LAST in forward dispatch, AFTER formation above (see
  // AssociationOutcome's comment for the LIFO reason). Only attempted when
  // formation did NOT fire this tick: the owner and a destination arriving
  // together is a coincidence (formation's job); the owner arriving ALONE is
  // recall's job. The two are mutually exclusive by construction.
  if (edge_bank_enabled && edge_bank_layout != nullptr && instance_enabled &&
      cloud_enabled && instance_receipt != nullptr &&
      cloud_receipt != nullptr && association_local.fired == 0u) {
    const SiteWord recall_owner_candidates =
        (instance_receipt->matched_mask | instance_receipt->recruited_mask) &
        grown_instance_basin_factor::kFullMask;
    if (__popc(recall_owner_candidates) == 1) {
      association_local.recall_attempted = 1u;
      const std::uint32_t basin = static_cast<std::uint32_t>(
          __ffs(static_cast<int>(recall_owner_candidates)) - 1);
      const std::uint64_t recall_owner_slot = instance_layout->rails
          [grown_instance_basin_factor::resident_index(
              grown_instance_basin_factor::basin_field(
                  basin, grown_instance_basin_factor::kAppearance))];
      const SiteWord recall_owner_token = words[recall_owner_slot];
      association_local.recall_owner_token = recall_owner_token;
      // Scan the edge bank on-device for an occupied record whose
      // owner_token matches -- the FIRST matching slot (lowest index) wins;
      // no host-supplied index ever selects the destination.
      std::uint32_t matched_edge = 0xffffffffu;
      SiteWord matched_dest = 0u;
      for (std::uint32_t edge = 0u; edge < resident_edge_bank::kEdgeCount;
           ++edge) {
        if (resident_edge_bank::read_field(
                words, *edge_bank_layout,
                resident_edge_bank::edge_field(
                    edge, resident_edge_bank::kOccupied)) != 0u &&
            resident_edge_bank::read_field(
                words, *edge_bank_layout,
                resident_edge_bank::edge_field(
                    edge, resident_edge_bank::kOwnerToken)) ==
                recall_owner_token) {
          matched_edge = edge;
          matched_dest = resident_edge_bank::read_field(
              words, *edge_bank_layout,
              resident_edge_bank::edge_field(edge,
                                             resident_edge_bank::kDestToken));
          break;
        }
      }
      if (matched_edge == 0xffffffffu) {
        // No edge record for this owner: abstain -- a required behaviour,
        // not a failure. Nothing is written.
        association_local.recall_abstained = 1u;
      } else {
        association_local.recall_edge_index = matched_edge;
        association_local.recall_dest_token = matched_dest;
        // Destination is derived from bank contents compared against the
        // owner token, entirely on device: exact_matching_cells() scans
        // every cloud cell's OWN cue rail for the recalled dest_token.
        const SiteWord dest_mask = grown_cloud_factor::exact_matching_cells(
            words, *cloud_layout, matched_dest);
        if (__popc(dest_mask) != 1) {
          // No cell (or an ambiguous multiple) carries this cue: abstain.
          association_local.recall_abstained = 1u;
        } else {
          const std::uint32_t cell = static_cast<std::uint32_t>(
              __ffs(static_cast<int>(dest_mask)) - 1);
          const std::uint64_t free_value_slot =
              grown_cloud_factor::fixed_physical_slot(
                  grown_cloud_factor::cell_rail(
                      cell, grown_cloud_factor::kCellFreeMatter, 0u));
          const std::uint64_t free_complement_slot =
              grown_cloud_factor::fixed_physical_slot(
                  grown_cloud_factor::cell_rail(
                      cell, grown_cloud_factor::kCellFreeMatter, 1u));
          const std::uint64_t support_value_slot =
              grown_cloud_factor::fixed_physical_slot(
                  grown_cloud_factor::cell_rail(
                      cell, grown_cloud_factor::kCellSupport, 0u));
          const std::uint64_t support_complement_slot =
              grown_cloud_factor::fixed_physical_slot(
                  grown_cloud_factor::cell_rail(
                      cell, grown_cloud_factor::kCellSupport, 1u));
          const SiteWord free_matter = words[free_value_slot];
          const SiteWord quantum = free_matter & (0u - free_matter);
          if (quantum == 0u) {
            // The recalled cell has no free matter left to move: abstain.
            association_local.recall_abstained = 1u;
          } else {
            // Destination-specific physical relocation: XOR is self-inverse
            // against a value/complement pair (same proof as
            // bcc32_resident_edge_bank.cuh), so recording (cell, quantum) is
            // all the inverse needs.
            words[free_value_slot] ^= quantum;
            words[free_complement_slot] ^= quantum;
            words[support_value_slot] ^= quantum;
            words[support_complement_slot] ^= quantum;
            association_local.recall_cell = cell;
            association_local.recall_quantum = quantum;
          }
        }
      }
    }
  }
  if (association_outcome != nullptr) *association_outcome = association_local;
  *advanced = ok ? (kResidentStepSucceeded |
                    (instance_advanced ? kResidentStepAdvancedInstance : 0u) |
                    (form_credit_step_mode ==
                             grown_form_credit_factor::kStepJournaled
                         ? kResidentStepJournaledFormCredit
                         : 0u) |
                    (form_credit_step_mode ==
                             grown_form_credit_factor::kStepEligibilityAge
                         ? kResidentStepAgedFormCredit
                         : 0u))
                 : 0u;
}

static __global__ void combined_resident_factors_inverse_kernel(
    SiteWord* words, bool cloud_enabled,
    const grown_cloud_factor::DeviceLayout* cloud_layout,
    bool instance_advanced,
    const grown_instance_basin_factor::DeviceLayout* instance_layout,
    bool sensorimotor_enabled,
    const grown_sensorimotor_factor::DeviceLayout* sensorimotor_layout,
    std::uint64_t raw_motor_zero_slot, std::uint64_t raw_motor_one_slot,
    bool readout_enabled,
    const resident_readout_f_route::DeviceLayout* readout_layout,
    bool form_credit_enabled, bool form_credit_journaled,
    bool form_credit_aged,
    const grown_form_credit_factor::DeviceLayout* form_credit_layout,
    bool edge_bank_enabled,
    const resident_edge_bank::DeviceLayout* edge_bank_layout,
    std::uint32_t association_edge_action,
    std::uint32_t association_edge_index, SiteWord association_owner_token,
    SiteWord association_dest_token, std::uint32_t association_recall_cell,
    SiteWord association_recall_quantum) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  // Recall's consequence (if any) is undone FIRST: it ran LAST in the
  // forward dispatch (after formation), so LIFO unwinding takes it first
  // here. Re-XORing the identical quantum into the identical cell's
  // free-matter/support pair restores the pre-recall state exactly --
  // association_recall_cell stays 0xffffffff whenever recall abstained (or
  // was never attempted), so there is nothing to undo in that case.
  if (cloud_enabled && association_recall_cell != 0xffffffffu) {
    const std::uint64_t free_value_slot =
        grown_cloud_factor::fixed_physical_slot(grown_cloud_factor::cell_rail(
            association_recall_cell, grown_cloud_factor::kCellFreeMatter,
            0u));
    const std::uint64_t free_complement_slot =
        grown_cloud_factor::fixed_physical_slot(grown_cloud_factor::cell_rail(
            association_recall_cell, grown_cloud_factor::kCellFreeMatter,
            1u));
    const std::uint64_t support_value_slot =
        grown_cloud_factor::fixed_physical_slot(grown_cloud_factor::cell_rail(
            association_recall_cell, grown_cloud_factor::kCellSupport, 0u));
    const std::uint64_t support_complement_slot =
        grown_cloud_factor::fixed_physical_slot(grown_cloud_factor::cell_rail(
            association_recall_cell, grown_cloud_factor::kCellSupport, 1u));
    words[free_value_slot] ^= association_recall_quantum;
    words[free_complement_slot] ^= association_recall_quantum;
    words[support_value_slot] ^= association_recall_quantum;
    words[support_complement_slot] ^= association_recall_quantum;
  }
  // Applied SECOND here because it was applied FIRST (of the two edge-bank-
  // adjacent steps) in the forward dispatch above (LIFO). Only a
  // kActionFormed outcome ever wrote anything (a
  // kActionDuplicate/kActionCapacityAbstained/kActionNone forward call was
  // already a no-op, so there is nothing to undo). The XOR write is its own
  // inverse: re-applying the identical (owner_token, dest_token) delta at the
  // identical edge_index restores the pre-formation ancilla state exactly.
  if (edge_bank_enabled && edge_bank_layout != nullptr &&
      association_edge_action == resident_edge_bank::kActionFormed) {
    resident_edge_bank::undo_edge_device(
        words, *edge_bank_layout, association_edge_index,
        association_owner_token, association_dest_token);
  }
  if (form_credit_journaled)
    grown_form_credit_factor::inverse_step_device(words, *form_credit_layout);
  else if (form_credit_aged)
    grown_form_credit_factor::inverse_age_eligibility(words,
                                                      *form_credit_layout);
  if (readout_enabled)
    resident_readout_f_route::inverse_step_device(words, *readout_layout);
  if (sensorimotor_enabled) {
    grown_sensorimotor_factor::inverse_step_device(words, *sensorimotor_layout);
    if (!form_credit_enabled) {
      project_sensorimotor_motor_surface(
          words,
          sensorimotor_layout->rails[grown_sensorimotor_factor::value_index(
              grown_sensorimotor_factor::kMotor)],
          raw_motor_zero_slot, raw_motor_one_slot);
    }
  }
  if (instance_advanced)
    grown_instance_basin_factor::restore_journal(words, *instance_layout);
  if (cloud_enabled)
    grown_cloud_factor::inverse_step_device(words, *cloud_layout);
}

#include "bcc32_developmental_adult_tail.inl"
