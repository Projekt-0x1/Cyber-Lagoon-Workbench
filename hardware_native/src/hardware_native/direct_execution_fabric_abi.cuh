#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_recipe_abi.cuh"
#include "hardware_native/direct_adult_legacy_oracle.cuh"

namespace substrate::direct_adult {

enum class DirectExecutionPhenotype : std::uint32_t {
  sparse_reference = 0u,
  packed_sparse = 1u,
  dense_tensor = 2u,
  tract_delay = 3u,
};

enum class DirectDenseNumericMode : std::uint32_t {
  int8_exact = 0u,
  fp16_guarded = 1u,
  bf16_guarded = 2u,
};

constexpr std::uint32_t kMaxTractDelay = 16u;
constexpr std::uint32_t kDenseTileDimension = 16u;
constexpr std::uint32_t kDenseTileWeightsCount = kDenseTileDimension * kDenseTileDimension;
constexpr std::uint32_t kExecutionFlagLesioned = 1u << 0;
constexpr std::uint32_t kExecutionFlagDirty = 1u << 1;
constexpr std::uint32_t kExecutionFlagFallbackGuard = 1u << 2;
constexpr std::uint32_t kLogicalMorphologyPlacementActive = 1u << 0;

// Hardware indirection only. These records contain no relation kind, label,
// concept, task, solver, or semantic dispatch key.
struct DirectLogicalMorphologyUnit {
  std::uint64_t logical_address;
  std::uint64_t state_identity;
  Word observable_state;
  std::uint32_t generation;
};
struct DirectLogicalMorphologyPlacement {
  std::uint64_t logical_address;
  std::uint64_t remap_epoch;
  std::uint32_t physical_slot;
  std::uint32_t multiprocessor;
  std::uint32_t storage_slot;
  std::uint32_t flags;
};
struct DirectLogicalMorphologyRemapCommand {
  std::uint64_t epoch;
  std::uint32_t physical_stride;
  std::uint32_t physical_offset;
  std::uint32_t physical_capacity;
  std::uint32_t multiprocessor_count;
};
static_assert(std::is_trivial_v<DirectLogicalMorphologyUnit> &&
              std::is_standard_layout_v<DirectLogicalMorphologyUnit> &&
              std::has_unique_object_representations_v<
                  DirectLogicalMorphologyUnit>);
static_assert(std::is_trivial_v<DirectLogicalMorphologyPlacement> &&
              std::is_standard_layout_v<DirectLogicalMorphologyPlacement> &&
              std::has_unique_object_representations_v<
                  DirectLogicalMorphologyPlacement>);
static_assert(std::is_trivial_v<DirectLogicalMorphologyRemapCommand> &&
              std::is_standard_layout_v<DirectLogicalMorphologyRemapCommand> &&
              std::has_unique_object_representations_v<
                  DirectLogicalMorphologyRemapCommand>);

struct DirectExecutionMembership {
  std::uint32_t phenotype;
  std::uint32_t owner_index;
  std::uint32_t local_slot;
  std::uint32_t reserved;
  std::uint64_t generation;
};
static_assert(std::is_trivial_v<DirectExecutionMembership> &&
              std::is_standard_layout_v<DirectExecutionMembership>);

struct DirectPackedSparsePanel {
  std::uint64_t generation;
  std::uint32_t source_begin;
  std::uint32_t source_count;
  std::uint32_t entry_begin;
  std::uint32_t entry_count;
  std::uint32_t revision_root;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectPackedSparsePanel> &&
              std::is_standard_layout_v<DirectPackedSparsePanel>);

struct DirectPackedEntry {
  std::uint32_t target;
  std::int32_t conductance_q16;
  std::uint32_t delay;
  std::uint32_t route_flags;
  std::uint64_t context_signature;
  std::uint32_t route_slot;
  // #1179: widened to 64-bit to match DirectRouteSlotMeta::generation exactly
  // (was truncated to 32 bits here) -- evaluate_packed_sparse_source now
  // checks this against the live slot's generation as a second, independent
  // fail-closed staleness guard alongside DirectPackedSourceMeta::source_revision,
  // so a physically reused route slot cannot be silently mistaken for the
  // stale cached entry that still names it.
  std::uint64_t route_generation;
  Word learned_output_word;
  std::uint16_t reserved16;
};
static_assert(std::is_trivial_v<DirectPackedEntry> &&
              std::is_standard_layout_v<DirectPackedEntry>);

struct DirectPackedSourceMeta {
  std::uint32_t entry_offset;
  std::uint32_t entry_count;
  std::uint32_t flags;
  // #1179: widened to 64-bit to match DirectNode::source_revision, which is
  // itself 64-bit precisely so this staleness guard cannot wrap under a
  // long-running adult. Snapshotting it into a 32-bit field here re-introduced
  // the wrap at the comparison, defeating that widening: after 2^32 membership
  // changes on one source the truncated snapshot would alias a live value and
  // the guard would read "current" on a stale panel. Ordered after `flags` to
  // keep the 8-byte member naturally aligned without an implicit pad.
  std::uint64_t source_revision;
};
static_assert(std::is_trivial_v<DirectPackedSourceMeta> &&
              std::is_standard_layout_v<DirectPackedSourceMeta>);

struct DirectDenseTile {
  std::uint64_t generation;
  std::uint32_t numeric_mode;
  std::uint32_t state_flags;
  std::uint32_t rows;
  std::uint32_t cols;
  std::uint32_t input_nodes[kDenseTileDimension];
  std::uint32_t output_nodes[kDenseTileDimension];
  std::int8_t weights_int8[kDenseTileWeightsCount];
  std::int32_t row_biases[kDenseTileDimension];
  std::int32_t threshold_q16;
  std::int32_t guard_band_q16;
  std::uint32_t activation_count;
};
static_assert(std::is_trivial_v<DirectDenseTile> &&
              std::is_standard_layout_v<DirectDenseTile>);

struct DirectTractLane {
  std::uint64_t generation;
  std::uint32_t source_node;
  std::uint32_t target_node;
  std::uint32_t delay;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectTractLane> &&
              std::is_standard_layout_v<DirectTractLane>);

struct DirectTractPacket {
  std::uint64_t sequence_key;
  ActivityEvent event;
  std::uint32_t due_tick;
  std::uint32_t lane_index;
  std::uint32_t original_event_ordinal;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectTractPacket> &&
              std::is_standard_layout_v<DirectTractPacket>);

}  // namespace substrate::direct_adult
