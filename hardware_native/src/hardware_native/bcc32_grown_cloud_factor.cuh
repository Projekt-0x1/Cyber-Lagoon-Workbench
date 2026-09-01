#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "hardware_native/bcc32_aperture_geometry.cuh"
#include "hardware_native/bcc32_types.cuh"

namespace substrate::bcc32::grown_cloud_factor {

using substrate::bcc32::SiteWord;

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

inline constexpr std::uint32_t kCellCount = 16u;
inline constexpr std::uint32_t kNeighborRadius = 6u;
inline constexpr std::uint32_t kNeighborCount = kNeighborRadius * 2u;
inline constexpr std::uint32_t kCellFieldCount = 14u;
inline constexpr std::uint32_t kRouteFieldCount = 6u;
inline constexpr std::uint32_t kHistoryDepth = 4u;
inline constexpr std::uint32_t kGlobalFieldCount = kHistoryDepth + 17u;
inline constexpr std::uint32_t kCellRailCount =
    kCellCount * kCellFieldCount * 2u;
inline constexpr std::uint32_t kRouteCount = kCellCount * kNeighborCount;
inline constexpr std::uint32_t kRouteRailCount =
    kRouteCount * kRouteFieldCount * 2u;
inline constexpr std::uint32_t kGlobalRailCount = kGlobalFieldCount * 2u;
inline constexpr std::uint32_t kResidentRailCount =
    kCellRailCount + kRouteRailCount + kGlobalRailCount;
inline constexpr std::uint32_t kJournalDepth = 128u;
inline constexpr std::uint32_t kJournalRailCount =
    kResidentRailCount * kJournalDepth;
inline constexpr std::uint32_t kRailCount =
    kResidentRailCount + kJournalRailCount;
inline constexpr std::uint32_t kMaxContactBytes = 192u;
inline constexpr std::uint32_t kActivationSimilarity = 23u;
inline constexpr SiteWord kCloudMarkerValue = 0xc10df27du;
inline constexpr SiteWord kCloudLayoutVersionValue = 0x6c617432u;

enum CellField : std::uint32_t {
  kCellOccupied = 0u,
  kCellCue = 1u,
  kCellSupport = 2u,
  kCellFreeMatter = 3u,
  kCellLesionEscrow = 4u,
  kCellPositiveResidual = 5u,
  kCellNegativeResidual = 6u,
  kCellPredictiveState = 7u,
  kCellFastCompetingSignature = 8u,
  kCellFastCompetingSupport = 9u,
  kCellDelayedCompetingSignature = 10u,
  kCellDelayedCompetingSupport = 11u,
  kCellCrossCompetingSignature = 12u,
  kCellCrossCompetingSupport = 13u,
};

enum RouteField : std::uint32_t {
  kFastWeight = 0u,
  kDelayedWeight = 1u,
  kCrossWeight = 2u,
  kFastEscrow = 3u,
  kDelayedEscrow = 4u,
  kCrossEscrow = 5u,
};

enum GlobalField : std::uint32_t {
  kHistory0 = 0u,
  kHistory1 = 1u,
  kHistory2 = 2u,
  kHistory3 = 3u,
  kTick = 4u,
  kPendingSignature = 5u,
  kPendingFlag = 6u,
  kLastSignature = 7u,
  kRecruitedSiteTotal = 8u,
  kRecruitedRouteTotal = 9u,
  kStrengthenedRouteTotal = 10u,
  kReleasedRouteTotal = 11u,
  kCloudMarker = 12u,
  kJournalCount = 13u,
  kJournalBeforeHashLow = 14u,
  kJournalBeforeHashHigh = 15u,
  kJournalMatterBefore = 16u,
  kPersistentBasinMask = 17u,
  kPersistentBasinState = 18u,
  kCloudLayoutVersion = 19u,
  kPersistentEligibility = 20u,
};

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kResidentRailCount]{};
};

// x,y are unchanged from the original placement (global x,y in [212,287]).
// z was relocated from base -36 (global [214,245]) to base 100 (global
// [350,412] at kJournalDepth=128) so a depth-128 journal -- needed because
// the largest real curriculum in the tree runs 123 develop() ticks against a
// journal that begin_journal() fills on every idle tick, not just contacts --
// fits without colliding with any other producer's full index domain
// (including journal ranges). Verified by a dedicated host-only enumeration
// probe over every producer's own physical_offset()-style function (the same
// discipline as the situation-tissue relocation in c2738b9b73): with the old
// z base, x/y already made cloud disjoint from every producer whose z range
// intersects [100,163] (situation, credit, credit-form, grounded-context all
// have z outside that band), so translating z alone -- stride and shape held
// fixed -- keeps the whole region inside the kSpatialMacroClosureRadius=26
// clearance window [26,473] on all three axes with no new collision.
__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {-38 + static_cast<std::int32_t>(index % 76u),
          -38 + static_cast<std::int32_t>((index / 76u) % 76u),
          100 + static_cast<std::int32_t>(index / (76u * 76u))};
}

struct ContactReceipt {
  std::uint64_t before_hash = 0u;
  std::uint64_t after_hash = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  SiteWord signature = 0u;
  SiteWord active_mask = 0u;
  SiteWord recruitment_anchor_mask = 0u;
  SiteWord recruited_mask = 0u;
  std::uint32_t recruited_sites = 0u;
  std::uint32_t recruited_routes = 0u;
  std::uint32_t strengthened_routes = 0u;
  std::uint32_t released_routes = 0u;
  std::uint32_t residual_bits = 0u;
  std::uint32_t residual_site_valid = 0u;
  std::uint32_t recovered_routes = 0u;
  std::uint32_t resource_before = 0u;
  std::uint32_t resource_after = 0u;
  SiteWord persistent_mask_before = 0u;
  SiteWord persistent_mask_after = 0u;
  SiteWord persistent_state_after = 0u;
  SiteWord persistent_eligibility_after = 0u;
};

struct PredictionWitness {
  SiteWord probe_signature = 0u;
  SiteWord predicted_signature = 0u;
  SiteWord history[kHistoryDepth]{};
  SiteWord surface_mask = 0u;
  SiteWord source_mask = 0u;
  SiteWord target_mask = 0u;
  SiteWord competing_target_mask = 0u;
  SiteWord route_source_mask = 0u;
  SiteWord predictive_state = 0u;
  SiteWord persistent_mask = 0u;
  SiteWord persistent_state = 0u;
  SiteWord persistent_eligibility = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t horizon = 0u;
  std::uint32_t support = 0u;
  std::uint32_t competing_support = 0u;
  std::uint32_t selected_field = 0u;
  std::uint32_t cross_population = 0u;
  std::uint32_t context_used = 0u;
  std::uint32_t ambiguous = 0u;
};

struct LesionReceipt {
  std::uint64_t before_hash = 0u;
  std::uint64_t after_hash = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  SiteWord source_mask = 0u;
  std::uint32_t moved_bits = 0u;
  std::uint32_t resource_before = 0u;
  std::uint32_t resource_after = 0u;
};

struct OhmWitness {
  SiteWord source_mask = 0u;
  SiteWord expected_mask = 0u;
  std::uint32_t expected_support = 0u;
  std::uint32_t competing_support = 0u;
  std::uint32_t selectivity_q8 = 0u;
};

struct ResourceWitness {
  std::uint32_t occupied_sites = 0u;
  std::uint32_t free_bits = 0u;
  std::uint32_t fast_bits = 0u;
  std::uint32_t delayed_bits = 0u;
  std::uint32_t cross_bits = 0u;
  std::uint32_t escrow_bits = 0u;
  std::uint32_t competing_bits = 0u;
  std::uint32_t journal_count = 0u;
};

struct TopologyWitness {
  std::uint64_t occupied_hash = 0u;
  std::uint64_t route_hash = 0u;
};

struct SituationWitness {
  SiteWord history[kHistoryDepth]{};
  SiteWord persistent_mask = 0u;
  SiteWord persistent_state = 0u;
  SiteWord persistent_eligibility = 0u;
  SiteWord occupied_mask = 0u;
  SiteWord exact_mask = 0u;
  SiteWord probe_signature = 0u;
  std::uint64_t before_hash = 0u;
  std::uint64_t after_hash = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
};

__host__ __device__ inline std::uint32_t cell_rail(
    std::uint32_t cell, std::uint32_t field, std::uint32_t polarity) {
  return (cell * kCellFieldCount + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t route_index(
    std::uint32_t source, std::uint32_t neighbor) {
  return source * kNeighborCount + neighbor;
}

__host__ __device__ inline std::uint32_t route_rail(
    std::uint32_t source, std::uint32_t neighbor, std::uint32_t field,
    std::uint32_t polarity) {
  return kCellRailCount +
         (route_index(source, neighbor) * kRouteFieldCount + field) * 2u +
         polarity;
}

__host__ __device__ inline std::uint32_t global_rail(
    std::uint32_t field, std::uint32_t polarity) {
  return kCellRailCount + kRouteRailCount + field * 2u + polarity;
}

__host__ __device__ inline std::uint32_t journal_rail(
    std::uint32_t event, std::uint32_t resident_rail) {
  return kResidentRailCount + event * kResidentRailCount + resident_rail;
}

__host__ __device__ inline std::uint64_t fixed_physical_slot(
    std::uint32_t index) {
  // edge_chunks, chunk_edge, chunk_sites, and centre are the shared aperture
  // geometry defined once in bcc32_aperture_geometry.cuh, which this file
  // can include without a cycle (bcc32_developmental_adult.cuh includes
  // this file, not the other way around).
  constexpr std::uint64_t edge_chunks =
      static_cast<std::uint64_t>(kApertureEdgeChunks);
  constexpr std::uint64_t chunk_edge = kApertureChunkEdge;
  constexpr std::uint64_t chunk_sites = kApertureChunkSites;
  constexpr std::int32_t centre = kApertureCentre;
  const PhysicalOffset offset = physical_offset(index);
  const std::uint64_t gx =
      static_cast<std::uint64_t>(centre + offset.x);
  const std::uint64_t gy =
      static_cast<std::uint64_t>(centre + offset.y);
  const std::uint64_t gz =
      static_cast<std::uint64_t>(centre + offset.z);
  const std::uint64_t chunk_x = gx / chunk_edge;
  const std::uint64_t chunk_y = gy / chunk_edge;
  const std::uint64_t chunk_z = gz / chunk_edge;
  const std::uint64_t chunk =
      (chunk_x * edge_chunks + chunk_y) * edge_chunks + chunk_z;
  const std::uint64_t local =
      ((gx % chunk_edge) * chunk_edge + (gy % chunk_edge)) * chunk_edge +
      (gz % chunk_edge);
  return chunk * chunk_sites + local;
}

__device__ inline std::uint64_t rail_slot(const DeviceLayout& layout,
                                          std::uint32_t rail) {
  return rail < kResidentRailCount ? layout.rails[rail]
                                   : fixed_physical_slot(rail);
}

__host__ __device__ inline std::uint32_t neighbor_target(
    std::uint32_t source, std::uint32_t neighbor) {
  const std::uint32_t distance = neighbor / 2u + 1u;
  if ((neighbor & 1u) == 0u) return (source + distance) % kCellCount;
  return (source + kCellCount - distance) % kCellCount;
}

__host__ __device__ inline std::uint32_t neighbor_slot(
    std::uint32_t source, std::uint32_t target) {
  for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
    if (neighbor_target(source, neighbor) == target) return neighbor;
  }
  return 0xffffffffu;
}

__device__ inline SiteWord read_pair(const SiteWord* words,
                                     const DeviceLayout& layout,
                                     std::uint32_t rail) {
  return words[rail_slot(layout, rail)];
}

__device__ inline void write_pair(SiteWord* words, const DeviceLayout& layout,
                                  std::uint32_t rail, SiteWord value) {
  words[rail_slot(layout, rail)] = value;
  words[rail_slot(layout, rail + 1u)] = ~value;
}

__device__ inline SiteWord read_cell(const SiteWord* words,
                                     const DeviceLayout& layout,
                                     std::uint32_t cell,
                                     std::uint32_t field) {
  return read_pair(words, layout, cell_rail(cell, field, 0u));
}

__device__ inline void write_cell(SiteWord* words,
                                  const DeviceLayout& layout,
                                  std::uint32_t cell, std::uint32_t field,
                                  SiteWord value) {
  write_pair(words, layout, cell_rail(cell, field, 0u), value);
}

__device__ inline SiteWord read_route(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t source,
                                      std::uint32_t neighbor,
                                      std::uint32_t field) {
  return read_pair(words, layout,
                   route_rail(source, neighbor, field, 0u));
}

__device__ inline void write_route(SiteWord* words,
                                   const DeviceLayout& layout,
                                   std::uint32_t source,
                                   std::uint32_t neighbor,
                                   std::uint32_t field, SiteWord value) {
  write_pair(words, layout, route_rail(source, neighbor, field, 0u), value);
}

__device__ inline SiteWord read_global(const SiteWord* words,
                                       const DeviceLayout& layout,
                                       std::uint32_t field) {
  return read_pair(words, layout, global_rail(field, 0u));
}

__device__ inline void write_global(SiteWord* words,
                                    const DeviceLayout& layout,
                                    std::uint32_t field, SiteWord value) {
  write_pair(words, layout, global_rail(field, 0u), value);
}

__host__ __device__ inline SiteWord mix32(SiteWord value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

__host__ __device__ inline SiteWord rotate_left(SiteWord value,
                                                std::uint32_t amount) {
  amount &= 31u;
  return amount == 0u ? value
                      : (value << amount) | (value >> (32u - amount));
}

__device__ inline SiteWord contact_signature(const std::uint8_t* bytes,
                                             std::uint32_t count) {
  std::int32_t votes[32]{};
  for (std::uint32_t index = 0u; index < count; ++index) {
    SiteWord feature = mix32(static_cast<SiteWord>(bytes[index]) +
                             0x9e3779b9u);
    if (index > 0u) {
      feature ^= mix32((static_cast<SiteWord>(bytes[index - 1u]) << 8u) |
                       bytes[index]);
    }
    if (index > 1u) {
      feature ^= rotate_left(
          mix32((static_cast<SiteWord>(bytes[index - 2u]) << 16u) |
                (static_cast<SiteWord>(bytes[index - 1u]) << 8u) |
                bytes[index]),
          index);
    }
    for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
      votes[bit] += ((feature >> bit) & 1u) != 0u ? 1 : -1;
    }
  }
  SiteWord result = 0u;
  for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
    if (votes[bit] >= 0) result |= 1u << bit;
  }
  return result;
}

__host__ __device__ inline std::uint32_t similarity(SiteWord left,
                                                    SiteWord right) {
#if defined(__CUDA_ARCH__)
  return 32u - static_cast<std::uint32_t>(__popc(left ^ right));
#else
  return 32u - static_cast<std::uint32_t>(
                   __builtin_popcount(left ^ right));
#endif
}

__device__ inline SiteWord increment_unary(SiteWord value) {
  const SiteWord empty = ~value;
  return empty == 0u ? value : value | (empty & (0u - empty));
}

__device__ inline SiteWord decrement_unary(SiteWord value) {
  return value == 0u ? 0u : value & (value - 1u);
}

__device__ inline std::uint64_t state_hash(const SiteWord* words,
                                           const DeviceLayout& layout) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::uint32_t index = 0u; index < kRailCount; ++index) {
    hash ^= words[rail_slot(layout, index)];
    hash *= 1099511628211ull;
  }
  return hash;
}

__device__ inline std::uint32_t matter_bits(const SiteWord* words,
                                            const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kRailCount; ++index) {
    total += static_cast<std::uint32_t>(
        __popc(words[rail_slot(layout, index)]));
  }
  return total;
}

__device__ inline bool begin_journal(SiteWord* words,
                                     const DeviceLayout& layout) {
  const std::uint32_t count =
      read_global(words, layout, kJournalCount);
  if (count >= kJournalDepth) return false;
  const std::uint64_t before_hash = state_hash(words, layout);
  const std::uint32_t matter_before = matter_bits(words, layout);
  for (std::uint32_t rail = 0u; rail < kResidentRailCount; rail += 2u) {
    write_pair(words, layout, journal_rail(count, rail),
               read_pair(words, layout, rail));
  }
  write_global(words, layout, kJournalCount, count + 1u);
  write_global(words, layout, kJournalBeforeHashLow,
               static_cast<SiteWord>(before_hash));
  write_global(words, layout, kJournalBeforeHashHigh,
               static_cast<SiteWord>(before_hash >> 32u));
  write_global(words, layout, kJournalMatterBefore, matter_before);
  return true;
}

__device__ inline bool journal_available(const SiteWord* words,
                                         const DeviceLayout& layout) {
  return read_global(words, layout, kJournalCount) < kJournalDepth;
}

__device__ inline std::uint64_t journal_before_hash(
    const SiteWord* words, const DeviceLayout& layout) {
  return static_cast<std::uint64_t>(
             read_global(words, layout, kJournalBeforeHashLow)) |
         (static_cast<std::uint64_t>(
              read_global(words, layout, kJournalBeforeHashHigh))
          << 32u);
}

__device__ inline void restore_last_journal(SiteWord* words,
                                            const DeviceLayout& layout) {
  const std::uint32_t count =
      read_global(words, layout, kJournalCount);
  if (count == 0u) return;
  const std::uint32_t event = count - 1u;
  for (std::uint32_t rail = 0u; rail < kResidentRailCount; rail += 2u) {
    const SiteWord previous =
        read_pair(words, layout, journal_rail(event, rail));
    write_pair(words, layout, rail, previous);
    write_pair(words, layout, journal_rail(event, rail), 0u);
  }
}

__device__ inline SiteWord matching_cells(const SiteWord* words,
                                          const DeviceLayout& layout,
                                          SiteWord signature) {
  SiteWord mask = 0u;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    if (read_cell(words, layout, cell, kCellOccupied) == 0u) continue;
    const std::uint32_t score =
        similarity(read_cell(words, layout, cell, kCellCue), signature);
    if (score < kActivationSimilarity) continue;
    mask |= 1u << cell;
  }
  return mask;
}

__device__ inline SiteWord exact_matching_cells(
    const SiteWord* words, const DeviceLayout& layout, SiteWord signature) {
  SiteWord mask = 0u;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    if (read_cell(words, layout, cell, kCellOccupied) == 0u) continue;
    if (read_cell(words, layout, cell, kCellCue) == signature) {
      mask |= 1u << cell;
    }
  }
  return mask;
}

__device__ inline std::uint32_t represented_resource_bits(
    const SiteWord* words, const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    total += static_cast<std::uint32_t>(
        __popc(read_cell(words, layout, cell, kCellOccupied)));
    total += static_cast<std::uint32_t>(
        __popc(read_cell(words, layout, cell, kCellFreeMatter)));
    total += static_cast<std::uint32_t>(
        __popc(read_cell(words, layout, cell, kCellLesionEscrow)));
    total += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellFastCompetingSupport)));
    total += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellDelayedCompetingSupport)));
    total += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellCrossCompetingSupport)));
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      for (std::uint32_t field = 0u; field < kRouteFieldCount; ++field) {
        total += static_cast<std::uint32_t>(
            __popc(read_route(words, layout, cell, neighbor, field)));
      }
    }
  }
  return total;
}

__device__ inline std::uint32_t first_empty_neighbor(
    const SiteWord* words, const DeviceLayout& layout, SiteWord anchors,
    std::uint32_t parity) {
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((anchors & (1u << source)) == 0u) continue;
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      const std::uint32_t target = neighbor_target(source, neighbor);
      if ((target & 1u) != parity) continue;
      if (read_cell(words, layout, target, kCellOccupied) == 0u) return target;
    }
  }
  return 0xffffffffu;
}

__device__ inline SiteWord recruit_pair(SiteWord* words,
                                        const DeviceLayout& layout,
                                        SiteWord signature,
                                        SiteWord anchors,
                                        std::uint32_t* recruited) {
  if (anchors == 0u) anchors = 1u;
  std::uint32_t fast = first_empty_neighbor(words, layout, anchors, 0u);
  std::uint32_t slow = first_empty_neighbor(words, layout, anchors, 1u);
  SiteWord mask = 0u;
  const std::uint32_t cells[2]{fast, slow};
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const std::uint32_t cell = cells[index];
    if (cell == 0xffffffffu) continue;
    SiteWord quantum = 0u;
    for (std::uint32_t source = 0u; source < kCellCount; ++source) {
      if ((anchors & (1u << source)) == 0u) continue;
      SiteWord free =
          read_cell(words, layout, source, kCellFreeMatter);
      if (free == 0u) continue;
      quantum = free & (0u - free);
      write_cell(words, layout, source, kCellFreeMatter, free & ~quantum);
      break;
    }
    if (quantum == 0u) continue;
    write_cell(words, layout, cell, kCellCue, signature);
    write_cell(words, layout, cell, kCellOccupied, quantum);
    write_cell(words, layout, cell, kCellSupport, 1u);
    mask |= 1u << cell;
    ++*recruited;
  }
  return mask;
}

__device__ inline std::uint32_t route_escrow_field(std::uint32_t field) {
  return field == kFastWeight
             ? kFastEscrow
             : (field == kDelayedWeight ? kDelayedEscrow : kCrossEscrow);
}

__device__ inline std::uint32_t competing_signature_field(
    std::uint32_t field) {
  return field == kFastWeight
             ? kCellFastCompetingSignature
             : (field == kDelayedWeight
                    ? kCellDelayedCompetingSignature
                    : kCellCrossCompetingSignature);
}

__device__ inline std::uint32_t competing_support_field(
    std::uint32_t field) {
  return field == kFastWeight
             ? kCellFastCompetingSupport
             : (field == kDelayedWeight
                    ? kCellDelayedCompetingSupport
                    : kCellCrossCompetingSupport);
}

__device__ inline bool strongest_prediction(
    const SiteWord* words, const DeviceLayout& layout, SiteWord sources,
    std::uint32_t field, SiteWord* signature, SiteWord* target_mask,
    std::uint32_t* support, SiteWord* route_source_mask = nullptr,
    std::uint32_t* cross_population = nullptr,
    SiteWord* competing_target_mask = nullptr,
    std::uint32_t* competing_support = nullptr,
    std::uint32_t* ambiguous = nullptr) {
  std::uint32_t best = 0u;
  SiteWord predicted = 0u;
  SiteWord targets = 0u;
  SiteWord selected_sources = 0u;
  std::uint32_t selected_cross = 0u;
  bool conflict = false;
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((sources & (1u << source)) == 0u) continue;
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      const SiteWord weight =
          read_route(words, layout, source, neighbor, field);
      const std::uint32_t candidate =
          static_cast<std::uint32_t>(__popc(weight));
      if (candidate == 0u) continue;
      const std::uint32_t target = neighbor_target(source, neighbor);
      const SiteWord cue = read_cell(words, layout, target, kCellCue);
      if (candidate > best) {
        best = candidate;
        predicted = cue;
        targets = 1u << target;
        selected_sources = 1u << source;
        selected_cross =
            field == kCrossWeight && (source & 1u) == 0u &&
                    (target & 1u) != 0u;
        conflict = false;
      } else if (candidate == best) {
        targets |= 1u << target;
        selected_sources |= 1u << source;
        selected_cross |=
            field == kCrossWeight && (source & 1u) == 0u &&
            (target & 1u) != 0u;
        conflict = conflict || cue != predicted;
      }
    }
  }
  SiteWord resident_competitors = 0u;
  std::uint32_t resident_competing_support = 0u;
  // Ungated: this scan must run even when best == 0u (the live route was
  // lesioned to zero support). A lesioned live route must not erase the
  // readout of a retained alternate — the M6 nested-binding contract's
  // direct per-cell scan proved a competing signature was still physically
  // present while this witness, when gated on `best != 0u`, reported
  // competing_target_mask = 0. When best == 0u, `predicted` is also 0u (it
  // is only ever assigned inside the `candidate > best` branch above, which
  // requires candidate > 0), so the skip condition below degenerates to
  // `competing_signature == 0u`, which is exactly the correct behaviour:
  // skip only cells with no competing signature at all.
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((sources & (1u << source)) == 0u) continue;
    const SiteWord competing_signature = read_cell(
        words, layout, source, competing_signature_field(field));
    if (competing_signature == 0u || competing_signature == predicted) {
      continue;
    }
    const std::uint32_t candidate = static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, source,
                  competing_support_field(field))));
    const SiteWord candidate_targets =
        matching_cells(words, layout, competing_signature);
    if (candidate > resident_competing_support) {
      resident_competing_support = candidate;
      resident_competitors = candidate_targets;
    } else if (candidate != 0u &&
               candidate == resident_competing_support) {
      resident_competitors |= candidate_targets;
    }
  }
  const bool resident_ambiguity =
      best != 0u && resident_competing_support >= best;
  if (signature != nullptr) *signature = predicted;
  if (target_mask != nullptr) *target_mask = targets;
  if (support != nullptr) *support = best;
  if (route_source_mask != nullptr) *route_source_mask = selected_sources;
  if (cross_population != nullptr) *cross_population = selected_cross;
  if (competing_target_mask != nullptr) {
    *competing_target_mask = resident_competitors;
  }
  if (competing_support != nullptr) {
    *competing_support = resident_competing_support;
  }
  if (ambiguous != nullptr) {
    *ambiguous = conflict || resident_ambiguity ? 1u : 0u;
  }
  return best != 0u && !conflict;
}

__device__ inline SiteWord predictive_state_for_cell(
    const SiteWord* words, const DeviceLayout& layout, std::uint32_t cell) {
  SiteWord fast = 0u;
  SiteWord cross = 0u;
  const SiteWord source = 1u << cell;
  if (!strongest_prediction(words, layout, source, kFastWeight, &fast,
                            nullptr, nullptr) ||
      !strongest_prediction(words, layout, source, kCrossWeight, &cross,
                            nullptr, nullptr)) {
    return 0u;
  }
  const SiteWord state =
      mix32(fast ^ 0x6d2b79f5u) ^
      rotate_left(mix32(cross ^ 0x9e3779b9u), 13u);
  return state == 0u ? 1u : state;
}

__device__ inline void refresh_predictive_states(
    SiteWord* words, const DeviceLayout& layout) {
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    const SiteWord occupied =
        read_cell(words, layout, cell, kCellOccupied);
    write_cell(words, layout, cell, kCellPredictiveState,
               occupied == 0u
                   ? 0u
                   : predictive_state_for_cell(words, layout, cell));
  }
}

__device__ inline SiteWord expand_predictive_basin(
    const SiteWord* words, const DeviceLayout& layout, SiteWord surface_mask,
    SiteWord* state_signature) {
  SiteWord state_set[kCellCount]{};
  std::uint32_t state_count = 0u;
  SiteWord reported_state = 0u;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    if ((surface_mask & (1u << cell)) == 0u) continue;
    const SiteWord state =
        read_cell(words, layout, cell, kCellPredictiveState);
    if (state == 0u) continue;
    bool seen = false;
    for (std::uint32_t index = 0u; index < state_count; ++index) {
      seen |= state_set[index] == state;
    }
    if (!seen) state_set[state_count++] = state;
    reported_state = reported_state == 0u ? state : reported_state;
  }
  SiteWord expanded = surface_mask;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    const SiteWord candidate =
        read_cell(words, layout, cell, kCellPredictiveState);
    for (std::uint32_t index = 0u; index < state_count; ++index) {
      if (candidate == state_set[index]) expanded |= 1u << cell;
    }
  }
  if (state_signature != nullptr) {
    *state_signature = state_count == 1u ? reported_state : 0u;
  }
  return expanded;
}

__device__ inline void learn_routes(
    SiteWord* words, const DeviceLayout& layout, SiteWord sources,
    SiteWord targets, std::uint32_t field, std::uint32_t* recruited,
    std::uint32_t* strengthened, std::uint32_t* released,
    std::uint32_t* recovered) {
  const std::uint32_t escrow_field = route_escrow_field(field);
  const std::uint32_t competing_signature =
      competing_signature_field(field);
  const std::uint32_t competing_support =
      competing_support_field(field);
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((sources & (1u << source)) == 0u) continue;
    bool learned = false;
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      const std::uint32_t target = neighbor_target(source, neighbor);
      SiteWord weight = read_route(words, layout, source, neighbor, field);
      if ((targets & (1u << target)) != 0u) {
        const bool was_empty = weight == 0u;
        SiteWord resident_competing_support =
            read_cell(words, layout, source, competing_support);
        if (resident_competing_support != 0u &&
            read_cell(words, layout, source, competing_signature) ==
                read_cell(words, layout, target, kCellCue)) {
          const SiteWord quantum = resident_competing_support &
                                   (0u - resident_competing_support);
          resident_competing_support &= ~quantum;
          weight |= quantum;
          write_cell(words, layout, source, competing_support,
                     resident_competing_support);
          if (resident_competing_support == 0u) {
            write_cell(words, layout, source, competing_signature, 0u);
          }
          write_route(words, layout, source, neighbor, field, weight);
          ++*recovered;
          learned = true;
          continue;
        }
        SiteWord escrow =
            read_route(words, layout, source, neighbor, escrow_field);
        if (escrow != 0u) {
          const SiteWord quantum = escrow & (0u - escrow);
          escrow &= ~quantum;
          weight |= quantum;
          write_route(words, layout, source, neighbor, escrow_field, escrow);
          write_route(words, layout, source, neighbor, field, weight);
          ++*recovered;
          learned = true;
        } else {
          SiteWord free =
              read_cell(words, layout, source, kCellFreeMatter);
          if (free == 0u &&
              resident_competing_support != 0u) {
            const SiteWord quantum = resident_competing_support &
                                     (0u - resident_competing_support);
            resident_competing_support &= ~quantum;
            write_cell(words, layout, source, competing_support,
                       resident_competing_support);
            if (resident_competing_support == 0u) {
              write_cell(words, layout, source, competing_signature, 0u);
            }
            free |= quantum;
          }
          if (free != 0u) {
            const SiteWord quantum = free & (0u - free);
            free &= ~quantum;
            weight |= quantum;
            write_cell(words, layout, source, kCellFreeMatter, free);
            write_route(words, layout, source, neighbor, field, weight);
            if (was_empty) ++*recruited;
            else ++*strengthened;
            learned = true;
          }
        }
      } else if (weight != 0u) {
        const SiteWord quantum = weight & (0u - weight);
        write_route(words, layout, source, neighbor, field,
                    weight & ~quantum);
        SiteWord old_support =
            read_cell(words, layout, source, competing_support);
        const SiteWord old_signature =
            read_cell(words, layout, source, competing_signature);
        const SiteWord target_signature =
            read_cell(words, layout, target, kCellCue);
        if (old_support != 0u && old_signature != target_signature) {
          write_cell(words, layout, source, kCellFreeMatter,
                     read_cell(words, layout, source, kCellFreeMatter) |
                         old_support);
          old_support = 0u;
        }
        write_cell(words, layout, source, competing_signature,
                   target_signature);
        write_cell(words, layout, source, competing_support,
                   old_support | quantum);
        ++*released;
      }
    }
    if (!learned) {
      // No nonlocal endpoint is installed. Local growth abstains rather than
      // smuggling a host-selected target into the resident topology.
    }
  }
}

__device__ inline ContactReceipt apply_contact(
    SiteWord* words, const DeviceLayout& layout, SiteWord signature) {
  ContactReceipt local{};
  local.before_hash = journal_before_hash(words, layout);
  local.matter_before =
      read_global(words, layout, kJournalMatterBefore);
  local.signature = signature;
  local.resource_before = represented_resource_bits(words, layout);
  local.persistent_mask_before =
      read_global(words, layout, kPersistentBasinMask);
  write_global(words, layout, kPersistentEligibility,
               read_global(words, layout, kPersistentEligibility) >> 1u);

  SiteWord active = matching_cells(words, layout, local.signature);
  const SiteWord anchors = read_global(words, layout, kHistory0);
  if (anchors != 0u) {
    local.residual_site_valid = 1u;
    SiteWord predicted = 0u;
    const bool valid = strongest_prediction(
        words, layout, anchors, kFastWeight, &predicted, nullptr, nullptr);
    const SiteWord positive = valid ? signature & ~predicted : signature;
    const SiteWord negative = valid ? predicted & ~signature : 0u;
    for (std::uint32_t source = 0u; source < kCellCount; ++source) {
      if ((anchors & (1u << source)) == 0u) continue;
      write_cell(words, layout, source, kCellPositiveResidual, positive);
      write_cell(words, layout, source, kCellNegativeResidual, negative);
    }
    local.residual_bits = static_cast<std::uint32_t>(
        __popc(positive | negative));
  }
  if (active == 0u) {
    const bool bootstrap =
        read_global(words, layout, kRecruitedSiteTotal) == 0u;
    SiteWord resident_residual = 0u;
    for (std::uint32_t source = 0u; source < kCellCount; ++source) {
      if ((anchors & (1u << source)) == 0u) continue;
      resident_residual |=
          read_cell(words, layout, source, kCellPositiveResidual);
      resident_residual |=
          read_cell(words, layout, source, kCellNegativeResidual);
    }
    if (bootstrap || resident_residual != 0u) {
      active = recruit_pair(words, layout, local.signature, anchors,
                            &local.recruited_sites);
      if (active != 0u) {
        local.recruitment_anchor_mask =
            anchors == 0u ? 1u : anchors;
        local.recruited_mask = active;
      }
    }
  }
  local.active_mask = active;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    if ((active & (1u << cell)) == 0u) continue;
    write_cell(words, layout, cell, kCellSupport,
               increment_unary(
                   read_cell(words, layout, cell, kCellSupport)));
  }

  const SiteWord age1 = read_global(words, layout, 0u);
  const SiteWord age3 = read_global(words, layout, 2u);
  learn_routes(words, layout, age1, active, kFastWeight,
               &local.recruited_routes, &local.strengthened_routes,
               &local.released_routes, &local.recovered_routes);
  learn_routes(words, layout, age3, active, kDelayedWeight,
               &local.recruited_routes, &local.strengthened_routes,
               &local.released_routes, &local.recovered_routes);

  SiteWord fast_sources = age1 & 0x5555u;
  SiteWord slow_targets = active & 0xaaaau;
  learn_routes(words, layout, fast_sources, slow_targets, kCrossWeight,
               &local.recruited_routes, &local.strengthened_routes,
               &local.released_routes, &local.recovered_routes);
  refresh_predictive_states(words, layout);
  SiteWord active_state = 0u;
  const SiteWord active_basin =
      expand_predictive_basin(words, layout, active, &active_state);
  if (active_state != 0u) {
    write_global(words, layout, kPersistentBasinMask, active_basin);
    write_global(words, layout, kPersistentBasinState, active_state);
    write_global(words, layout, kPersistentEligibility, 0xffffffffu);
  }

  for (std::uint32_t age = kHistoryDepth - 1u; age > 0u; --age) {
    write_global(words, layout, age,
                 read_global(words, layout, age - 1u));
  }
  write_global(words, layout, 0u, active);
  write_global(words, layout, kTick,
               read_global(words, layout, kTick) + 1u);
  write_global(words, layout, kLastSignature, signature);
  write_global(words, layout, kRecruitedSiteTotal,
               read_global(words, layout, kRecruitedSiteTotal) +
                   local.recruited_sites);
  write_global(words, layout, kRecruitedRouteTotal,
               read_global(words, layout, kRecruitedRouteTotal) +
                   local.recruited_routes);
  write_global(words, layout, kStrengthenedRouteTotal,
               read_global(words, layout, kStrengthenedRouteTotal) +
                   local.strengthened_routes);
  write_global(words, layout, kReleasedRouteTotal,
               read_global(words, layout, kReleasedRouteTotal) +
                   local.released_routes);

  local.after_hash = state_hash(words, layout);
  local.matter_after = matter_bits(words, layout);
  local.resource_after = represented_resource_bits(words, layout);
  local.persistent_mask_after =
      read_global(words, layout, kPersistentBasinMask);
  local.persistent_state_after =
      read_global(words, layout, kPersistentBasinState);
  local.persistent_eligibility_after =
      read_global(words, layout, kPersistentEligibility);
  return local;
}

static __global__ void stage_contact_kernel(
    SiteWord* words, DeviceLayout layout, const std::uint8_t* bytes,
    std::uint32_t count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (read_global(words, layout, kPendingFlag) == 0u &&
      !begin_journal(words, layout)) {
    return;
  }
  write_global(words, layout, kPendingSignature,
               contact_signature(bytes, count));
  write_global(words, layout, kPendingFlag, 1u);
}

__device__ inline bool step_device(SiteWord* words, DeviceLayout layout,
                                   ContactReceipt* receipt) {
  ContactReceipt local{};
  if (read_global(words, layout, kPendingFlag) != 0u) {
    local = apply_contact(
        words, layout, read_global(words, layout, kPendingSignature));
    write_global(words, layout, kPendingFlag, 0u);
    local.after_hash = state_hash(words, layout);
    local.matter_after = matter_bits(words, layout);
    local.resource_after = represented_resource_bits(words, layout);
  } else {
    if (!begin_journal(words, layout)) return false;
    local.before_hash = journal_before_hash(words, layout);
    local.matter_before =
        read_global(words, layout, kJournalMatterBefore);
    for (std::uint32_t age = kHistoryDepth - 1u; age > 0u; --age) {
      write_global(words, layout, age,
                   read_global(words, layout, age - 1u));
    }
    write_global(words, layout, kHistory0, 0u);
    write_global(words, layout, kPersistentEligibility,
                 read_global(words, layout, kPersistentEligibility) >> 1u);
    write_global(words, layout, kTick,
                 read_global(words, layout, kTick) + 1u);
    local.after_hash = state_hash(words, layout);
    local.matter_after = matter_bits(words, layout);
  }
  if (receipt != nullptr) *receipt = local;
  return true;
}

static __global__ void step_kernel(SiteWord* words, DeviceLayout layout,
                                   ContactReceipt* receipt,
                                   std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *advanced = step_device(words, layout, receipt) ? 1u : 0u;
}

static __global__ void contact_kernel(
    SiteWord* words, DeviceLayout layout, const std::uint8_t* bytes,
    std::uint32_t count, ContactReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (!begin_journal(words, layout)) return;
  ContactReceipt local =
      apply_contact(words, layout, contact_signature(bytes, count));
  local.after_hash = state_hash(words, layout);
  local.matter_after = matter_bits(words, layout);
  if (receipt != nullptr) *receipt = local;
}

__device__ inline void inverse_step_device(SiteWord* words,
                                           DeviceLayout layout) {
  restore_last_journal(words, layout);
}

static __global__ void inverse_step_kernel(SiteWord* words,
                                           DeviceLayout layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  inverse_step_device(words, layout);
}

static __global__ void predict_kernel(
    const SiteWord* words, DeviceLayout layout, const std::uint8_t* bytes,
    std::uint32_t count, std::uint32_t horizon,
    PredictionWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  PredictionWitness local{};
  local.horizon = horizon;
  local.probe_signature = contact_signature(bytes, count);
  for (std::uint32_t age = 0u; age < kHistoryDepth; ++age) {
    local.history[age] = read_global(words, layout, age);
  }
  local.surface_mask = matching_cells(words, layout, local.probe_signature);
  const SiteWord expanded_source =
      expand_predictive_basin(words, layout, local.surface_mask,
                              &local.predictive_state);
  SiteWord surface_delayed = 0u;
  bool surface_has_delayed =
      strongest_prediction(words, layout, local.surface_mask,
                           kDelayedWeight, &surface_delayed, nullptr, nullptr);
  if (!surface_has_delayed) {
    for (std::uint32_t source = 0u;
         source < kCellCount && !surface_has_delayed; ++source) {
      if ((local.surface_mask & (1u << source)) == 0u) continue;
      for (std::uint32_t neighbor = 0u;
           neighbor < kNeighborCount; ++neighbor) {
        surface_has_delayed =
            read_route(words, layout, source, neighbor,
                       route_escrow_field(kDelayedWeight)) != 0u;
        if (surface_has_delayed) break;
      }
    }
  }
  local.source_mask =
      surface_has_delayed ? expanded_source : local.surface_mask;
  local.persistent_mask =
      read_global(words, layout, kPersistentBasinMask);
  local.persistent_state =
      read_global(words, layout, kPersistentBasinState);
  local.persistent_eligibility =
      read_global(words, layout, kPersistentEligibility);
  const std::uint32_t field =
      horizon <= 1u ? kFastWeight
                    : (horizon == 2u ? kCrossWeight : kDelayedWeight);
  local.selected_field = field;
  SiteWord direct_prediction = 0u;
  std::uint32_t direct_support = 0u;
  std::uint32_t direct_ambiguous = 0u;
  const bool direct_valid = strongest_prediction(
      words, layout, local.source_mask, field, &direct_prediction, nullptr,
      &direct_support, nullptr, nullptr, nullptr, nullptr,
      &direct_ambiguous);
  SiteWord expected_probe = 0u;
  const bool context_predicts_probe =
      local.persistent_mask != 0u &&
      local.persistent_eligibility != 0u &&
      strongest_prediction(words, layout, local.persistent_mask,
                           kFastWeight, &expected_probe, nullptr, nullptr);
  SiteWord expected_state = 0u;
  SiteWord expected_surface = 0u;
  if (context_predicts_probe) {
    expected_surface = matching_cells(words, layout, expected_probe);
    (void)expand_predictive_basin(
        words, layout, expected_surface, &expected_state);
  }
  const bool context_links_probe =
      context_predicts_probe &&
      (similarity(expected_probe, local.probe_signature) >=
           kActivationSimilarity ||
       (expected_state != 0u &&
        expected_state == local.predictive_state));
  SiteWord resolved_prediction = 0u;
  std::uint32_t resolved_support = 0u;
  const bool resolved_valid =
      context_links_probe &&
      strongest_prediction(words, layout, expected_surface, field,
                           &resolved_prediction, nullptr, &resolved_support,
                           nullptr, nullptr);
  const bool resolved_wins =
      resolved_valid &&
      (!direct_valid || direct_ambiguous != 0u ||
       resolved_support > direct_support);
  if (context_links_probe && resolved_wins) {
    local.source_mask = expected_surface;
    local.predictive_state = expected_state;
    local.context_used = 1u;
  }
  if (strongest_prediction(words, layout, local.source_mask, field,
                           &local.predicted_signature, &local.target_mask,
                           &local.support, &local.route_source_mask,
                           &local.cross_population,
                           &local.competing_target_mask,
                           &local.competing_support, &local.ambiguous)) {
    local.valid = 1u;
  }
  if (witness != nullptr) *witness = local;
}

static __global__ void ohm_kernel(
    const SiteWord* words, DeviceLayout layout,
    const std::uint8_t* probe_bytes, std::uint32_t probe_count,
    const std::uint8_t* expected_bytes, std::uint32_t expected_count,
    std::uint32_t horizon, OhmWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  OhmWitness local{};
  local.source_mask = expand_predictive_basin(
      words, layout,
      matching_cells(words, layout,
                     contact_signature(probe_bytes, probe_count)),
      nullptr);
  local.expected_mask = matching_cells(
      words, layout, contact_signature(expected_bytes, expected_count));
  const std::uint32_t field =
      horizon <= 1u ? kFastWeight
                    : (horizon == 2u ? kCrossWeight : kDelayedWeight);
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((local.source_mask & (1u << source)) == 0u) continue;
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      const std::uint32_t target = neighbor_target(source, neighbor);
      const std::uint32_t support = static_cast<std::uint32_t>(
          __popc(read_route(words, layout, source, neighbor, field)));
      if ((local.expected_mask & (1u << target)) != 0u) {
        local.expected_support += support;
      } else {
        local.competing_support += support;
      }
    }
  }
  const std::uint32_t total =
      local.expected_support + local.competing_support;
  local.selectivity_q8 =
      total == 0u ? 0u : (local.expected_support * 255u) / total;
  if (witness != nullptr) *witness = local;
}

static __global__ void resource_kernel(
    const SiteWord* words, DeviceLayout layout, ResourceWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  ResourceWitness local{};
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    local.occupied_sites +=
        read_cell(words, layout, cell, kCellOccupied) != 0u;
    local.free_bits += static_cast<std::uint32_t>(
        __popc(read_cell(words, layout, cell, kCellFreeMatter)));
    local.competing_bits += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellFastCompetingSupport)));
    local.competing_bits += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellDelayedCompetingSupport)));
    local.competing_bits += static_cast<std::uint32_t>(__popc(
        read_cell(words, layout, cell, kCellCrossCompetingSupport)));
    for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount; ++neighbor) {
      local.fast_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kFastWeight)));
      local.delayed_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kDelayedWeight)));
      local.cross_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kCrossWeight)));
      local.escrow_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kFastEscrow)));
      local.escrow_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kDelayedEscrow)));
      local.escrow_bits += static_cast<std::uint32_t>(
          __popc(read_route(words, layout, cell, neighbor, kCrossEscrow)));
    }
  }
  local.journal_count = read_global(words, layout, kJournalCount);
  if (witness != nullptr) *witness = local;
}

static __global__ void topology_kernel(
    const SiteWord* words, DeviceLayout layout, TopologyWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  TopologyWitness local{1469598103934665603ull, 1469598103934665603ull};
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    local.occupied_hash ^=
        read_cell(words, layout, cell, kCellOccupied) != 0u ? cell + 1u : 0u;
    local.occupied_hash *= 1099511628211ull;
  }
  for (std::uint32_t route = 0u; route < kRouteCount; ++route) {
    const std::uint32_t source = route / kNeighborCount;
    const std::uint32_t neighbor = route % kNeighborCount;
    for (std::uint32_t field = kFastWeight; field <= kCrossWeight; ++field) {
      local.route_hash ^=
          read_route(words, layout, source, neighbor, field) != 0u
              ? (route + 1u) * (field + 1u)
              : 0u;
      local.route_hash *= 1099511628211ull;
    }
  }
  if (witness != nullptr) *witness = local;
}

static __global__ void situation_kernel(
    const SiteWord* words, DeviceLayout layout, const std::uint8_t* bytes,
    std::uint32_t count, SituationWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  SituationWitness local{};
  local.before_hash = state_hash(words, layout);
  local.matter_before = matter_bits(words, layout);
  for (std::uint32_t age = 0u; age < kHistoryDepth; ++age) {
    local.history[age] = read_global(words, layout, age);
  }
  local.persistent_mask =
      read_global(words, layout, kPersistentBasinMask);
  local.persistent_state =
      read_global(words, layout, kPersistentBasinState);
  local.persistent_eligibility =
      read_global(words, layout, kPersistentEligibility);
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    if (read_cell(words, layout, cell, kCellOccupied) != 0u) {
      local.occupied_mask |= 1u << cell;
    }
  }
  if (bytes != nullptr && count != 0u) {
    local.probe_signature = contact_signature(bytes, count);
    local.exact_mask =
        exact_matching_cells(words, layout, local.probe_signature);
  }
  local.after_hash = state_hash(words, layout);
  local.matter_after = matter_bits(words, layout);
  if (witness != nullptr) *witness = local;
}

static __global__ void lesion_kernel(
    SiteWord* words, DeviceLayout layout, SiteWord source_mask,
    std::uint32_t family, std::uint32_t max_bits, std::uint32_t restore,
    LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  LesionReceipt local{};
  local.before_hash = state_hash(words, layout);
  local.matter_before = matter_bits(words, layout);
  local.source_mask = source_mask;
  local.resource_before = represented_resource_bits(words, layout);
  const std::uint32_t first_weight =
      family == 4u ? kFastWeight
                   : (family == 0u ? kFastWeight
                   : (family == 1u ? kDelayedWeight
                                   : (family == 2u ? kCrossWeight
                                                  : kFastWeight)));
  const std::uint32_t last_weight =
      family == 3u ? kCrossWeight : first_weight;
  for (std::uint32_t source = 0u; source < kCellCount; ++source) {
    if ((source_mask & (1u << source)) == 0u) continue;
    if (family == 4u) {
      const SiteWord occupied =
          read_cell(words, layout, source, kCellOccupied);
      const SiteWord escrow =
          read_cell(words, layout, source, kCellLesionEscrow);
      if (restore != 0u) {
        write_cell(words, layout, source, kCellOccupied,
                   occupied | escrow);
        write_cell(words, layout, source, kCellLesionEscrow, 0u);
        local.moved_bits +=
            static_cast<std::uint32_t>(__popc(escrow));
      } else {
        SiteWord retained = occupied;
        SiteWord moved = 0u;
        while (retained != 0u && local.moved_bits < max_bits) {
          const SiteWord quantum = retained & (0u - retained);
          retained &= ~quantum;
          moved |= quantum;
          ++local.moved_bits;
        }
        write_cell(words, layout, source, kCellOccupied, retained);
        write_cell(words, layout, source, kCellLesionEscrow,
                   escrow | moved);
      }
      continue;
    }
    for (std::uint32_t weight_field = first_weight;
         weight_field <= last_weight; ++weight_field) {
      const std::uint32_t escrow_field =
          route_escrow_field(weight_field);
      for (std::uint32_t neighbor = 0u; neighbor < kNeighborCount;
           ++neighbor) {
        const SiteWord weight =
            read_route(words, layout, source, neighbor, weight_field);
        const SiteWord escrow =
            read_route(words, layout, source, neighbor, escrow_field);
        if (restore != 0u) {
          write_route(words, layout, source, neighbor, weight_field,
                      weight | escrow);
          write_route(words, layout, source, neighbor, escrow_field, 0u);
          local.moved_bits += static_cast<std::uint32_t>(__popc(escrow));
          continue;
        }
        SiteWord retained = weight;
        SiteWord moved = 0u;
        while (retained != 0u && local.moved_bits < max_bits) {
          const SiteWord quantum = retained & (0u - retained);
          retained &= ~quantum;
          moved |= quantum;
          ++local.moved_bits;
        }
        write_route(words, layout, source, neighbor, weight_field, retained);
        write_route(words, layout, source, neighbor, escrow_field,
                    escrow | moved);
      }
    }
  }
  local.after_hash = state_hash(words, layout);
  local.matter_after = matter_bits(words, layout);
  local.resource_after = represented_resource_bits(words, layout);
  if (receipt != nullptr) *receipt = local;
}

}  // namespace substrate::bcc32::grown_cloud_factor
