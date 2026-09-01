#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "hardware_native/direct_execution_fabric_abi.cuh"
#include "hardware_native/direct_dynamic_topology_arena.cuh"
#include "hardware_native/direct_adult_legacy_oracle.cuh"
namespace substrate::direct_adult {
constexpr std::int32_t kConductanceFloorQ16 = 1 << 12;
constexpr std::uint32_t kEligibilityLifetime = 4u;
__host__ __device__ inline std::int32_t clamp_conductance(std::int64_t value) {
  if (value < kConductanceFloorQ16)
    return kConductanceFloorQ16;
  if (value > 4ll * kConductanceOneQ16)
    return static_cast<std::int32_t>(4ll * kConductanceOneQ16);
  return static_cast<std::int32_t>(value);
}
__host__ __device__ inline std::uint32_t mix_signature(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  const std::uint32_t folded = static_cast<std::uint32_t>(value ^ (value >> 32));
  return folded == 0u ? 1u : folded;
}
__host__ __device__ inline std::uint64_t first_contact_context_signature(Word word) {
  return mix_signature(static_cast<std::uint64_t>(word) * 0x517cc1b727220a95ull);
}

struct DirectCanonicalTransitionResult;

struct DirectLogicalMorphologyFabricDeviceView {
  DirectLogicalMorphologyUnit* units;
  DirectLogicalMorphologyPlacement* placements;
  std::uint32_t* count;
  std::uint64_t* remap_epoch;
  std::uint32_t capacity;
  std::uint32_t physical_capacity;
  std::uint32_t multiprocessor_count;
  std::uint32_t declared_bytes;
};

__host__ __device__ inline std::uint32_t logical_morphology_required_bytes(
    std::uint32_t capacity) {
  const std::uint64_t bytes = sizeof(std::uint32_t) + sizeof(std::uint64_t) +
      static_cast<std::uint64_t>(capacity) *
          (sizeof(DirectLogicalMorphologyUnit) +
           sizeof(DirectLogicalMorphologyPlacement));
  return bytes > 0xffffffffull ? 0u : static_cast<std::uint32_t>(bytes);
}

__device__ inline std::uint32_t logical_morphology_gcd(
    std::uint32_t left, std::uint32_t right) {
  while (right != 0u) {
    const std::uint32_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

__device__ inline bool logical_morphology_view_valid(
    const DirectLogicalMorphologyFabricDeviceView& fabric) {
  return fabric.units != nullptr && fabric.placements != nullptr &&
      fabric.count != nullptr && fabric.remap_epoch != nullptr &&
      fabric.capacity != 0u && fabric.physical_capacity >= fabric.capacity &&
      fabric.multiprocessor_count != 0u &&
      fabric.declared_bytes == logical_morphology_required_bytes(fabric.capacity);
}

__device__ inline bool admit_logical_morphology_unit(
    const DirectLogicalMorphologyFabricDeviceView& fabric,
    const DirectLogicalMorphologyUnit& unit) {
  if (!logical_morphology_view_valid(fabric) || unit.logical_address == 0u ||
      unit.state_identity == 0u || unit.generation == 0u ||
      *fabric.count >= fabric.capacity) return false;
  for (std::uint32_t i = 0u; i < *fabric.count; ++i)
    if (fabric.units[i].logical_address == unit.logical_address) return false;
  const std::uint32_t slot = *fabric.count;
  fabric.units[slot] = unit;
  fabric.placements[slot] = DirectLogicalMorphologyPlacement{
      unit.logical_address, *fabric.remap_epoch, slot,
      slot % fabric.multiprocessor_count, slot,
      kLogicalMorphologyPlacementActive};
  __threadfence();
  *fabric.count = slot + 1u;
  return true;
}

__device__ inline const DirectLogicalMorphologyUnit*
resolve_logical_morphology_unit(
    const DirectLogicalMorphologyFabricDeviceView& fabric,
    std::uint64_t logical_address,
    DirectLogicalMorphologyPlacement* placement = nullptr) {
  if (!logical_morphology_view_valid(fabric) || logical_address == 0u ||
      *fabric.count > fabric.capacity) return nullptr;
  for (std::uint32_t i = 0u; i < *fabric.count; ++i) {
    const auto& unit = fabric.units[i];
    const auto& mapped = fabric.placements[i];
    if (unit.logical_address != logical_address) continue;
    if (mapped.logical_address != logical_address || mapped.storage_slot != i ||
        mapped.physical_slot >= fabric.physical_capacity ||
        mapped.multiprocessor >= fabric.multiprocessor_count ||
        mapped.flags != kLogicalMorphologyPlacementActive) return nullptr;
    if (placement != nullptr) *placement = mapped;
    return &unit;
  }
  return nullptr;
}

__device__ inline bool remap_logical_morphology_fabric(
    const DirectLogicalMorphologyFabricDeviceView& fabric,
    const DirectLogicalMorphologyRemapCommand& command) {
  if (!logical_morphology_view_valid(fabric) || command.epoch <= *fabric.remap_epoch ||
      command.physical_capacity != fabric.physical_capacity ||
      command.multiprocessor_count != fabric.multiprocessor_count ||
      command.physical_stride == 0u ||
      logical_morphology_gcd(command.physical_stride,
                             fabric.physical_capacity) != 1u ||
      *fabric.count > fabric.capacity) return false;
  for (std::uint32_t i = 0u; i < *fabric.count; ++i) {
    auto& placement = fabric.placements[i];
    const std::uint32_t physical =
        (i * command.physical_stride + command.physical_offset) %
        fabric.physical_capacity;
    placement.logical_address = fabric.units[i].logical_address;
    placement.remap_epoch = command.epoch;
    placement.physical_slot = physical;
    placement.multiprocessor = physical % fabric.multiprocessor_count;
    placement.storage_slot = i;
    placement.flags = kLogicalMorphologyPlacementActive;
  }
  __threadfence();
  *fabric.remap_epoch = command.epoch;
  return true;
}

struct DirectExecutionFabricDeviceView {
  DirectExecutionMembership* node_memberships;
  DirectExecutionMembership* route_memberships;
  DirectPackedSparsePanel* packed_panels;
  DirectPackedSourceMeta* packed_source_meta;
  DirectPackedEntry* packed_entries;
  DirectDenseTile* dense_tiles;
  DirectTractLane* tract_lanes;
  DirectTractPacket* tract_ring_packets;
  std::uint32_t* tract_bucket_counts;
  std::uint32_t max_tract_delay;

  ActivityEvent* staged_events;
  std::uint32_t* staged_event_valid;
  std::uint32_t* staged_event_ranks;
  std::uint32_t* staged_event_total;
};

struct DirectExecutionFabricRuntime {
  DirectExecutionFabricDeviceView view;

  std::uint32_t node_count;
  std::uint32_t route_capacity;
  std::uint32_t frontier_capacity;

  std::uint32_t packed_panel_count;
  std::uint32_t packed_panel_capacity;
  std::uint32_t packed_entry_count;
  std::uint32_t packed_entry_capacity;

  std::uint32_t dense_tile_count;
  std::uint32_t dense_tile_capacity;

  std::uint32_t tract_lane_count;
  std::uint32_t tract_lane_capacity;
  std::uint32_t max_tract_delay;

  std::uint64_t generation_epoch;

  void* scan_storage;
  std::size_t scan_storage_bytes;

  std::uint64_t* tract_sort_keys_in;
  std::uint64_t* tract_sort_keys_out;
  std::uint32_t* tract_sort_values_in;
  std::uint32_t* tract_sort_values_out;
  void* tract_sort_storage;
  std::size_t tract_sort_storage_bytes;

  // #1208: sticky, host-side. See launch_direct_heterogeneous_frontier_step.
  bool tract_ring_ever_written;
};

DirectExecutionFabricRuntime* create_direct_execution_fabric(
    std::uint32_t node_count, std::uint32_t route_capacity,
    std::uint32_t frontier_capacity, std::uint32_t max_tract_delay = kMaxTractDelay);

void destroy_direct_execution_fabric(DirectExecutionFabricRuntime* fabric);

bool install_direct_packed_sparse_panel(
    DirectBrainV01* brain, DirectExecutionFabricRuntime* fabric,
    std::uint32_t source_begin, std::uint32_t source_count);

bool refresh_direct_packed_sparse_source(
    const DirectBrainV01& brain, DirectExecutionFabricRuntime* fabric,
    std::uint32_t source);

bool install_direct_dense_tile(
    DirectBrainV01* brain, DirectExecutionFabricRuntime* fabric,
    const DirectDenseTile& tile);

bool install_direct_tract_lane(
    DirectBrainV01* brain, DirectExecutionFabricRuntime* fabric,
    std::uint32_t source, std::uint32_t target, std::uint32_t delay);

void launch_direct_heterogeneous_frontier_step(
    DirectAdultRuntime* runtime, std::uint32_t frontier_work);

std::uint32_t get_pending_tract_packet_count(
    const DirectExecutionFabricRuntime& fabric);

// #1236: packed sparse is candidate discovery only. It returns the complete
// canonical transition result; motor/eligibility/materialization/commit are
// derived centrally by the heterogeneous dispatcher rather than being hidden
// inside a backend-specific one-output path.
__device__ bool evaluate_packed_sparse_source(
    DirectBrainV01 brain,
    const DirectExecutionFabricDeviceView& fabric,
    std::uint32_t panel_index,
    const ActivityEvent& event,
    DirectCanonicalTransitionResult* out_result,
    AdultCounters* counters);

__device__ bool execute_dense_tensor_tile(
    const DirectBrainV01& brain,
    const DirectExecutionFabricDeviceView& fabric,
    std::uint32_t tile_index,
    const ActivityEvent& event,
    ActivityEvent* out_event,
    AdultCounters* counters);

void launch_wmma_dense_tensor_tile_eval(
    const DirectDenseTile* d_tile,
    const std::int8_t* d_in,
    std::int32_t* d_out,
    std::uint32_t tile_count,
    bool fp16 = false);

__global__ void enqueue_tract_packet_kernel(
    DirectExecutionFabricDeviceView fabric,
    DirectTractPacket packet,
    std::uint32_t stride);

// #1327/#1208: the tract ring's only writer, and the only place that may mark
// the ring dirty. See direct_tract_execution.cu.
void enqueue_direct_tract_packet(DirectExecutionFabricRuntime* fabric,
                                 const DirectTractPacket& packet);

__global__ void gather_tract_sort_keys_kernel(
    const DirectTractPacket* bucket_packets,
    std::uint32_t bucket_count,
    std::uint64_t* keys_out,
    std::uint32_t* values_out);

__global__ void deliver_sorted_tract_packets_kernel(
    const DirectTractPacket* bucket_packets,
    const std::uint32_t* sorted_indices,
    std::uint32_t bucket_count,
    ActivityEvent* staged_events,
    std::uint32_t* staged_event_valid,
    std::uint32_t base_offset,
    AdultCounters* counters);

__global__ void clear_tract_bucket_count_kernel(
    std::uint32_t* bucket_counts,
    std::uint32_t bucket);


}  // namespace substrate::direct_adult
