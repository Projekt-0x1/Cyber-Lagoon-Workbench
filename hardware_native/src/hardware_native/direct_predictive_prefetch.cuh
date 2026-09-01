#ifndef HARDWARE_NATIVE_DIRECT_PREDICTIVE_PREFETCH_CUH
#define HARDWARE_NATIVE_DIRECT_PREDICTIVE_PREFETCH_CUH

#include <cstdint>

#include "hardware_native/direct_exact_history_cold_archive.cuh"

namespace substrate::direct_network {

// Anticipatory staging of warm/cold memory blocks driven by prospective
// shadow rollouts.  Prefetch is a performance nomination and nothing else:
// the rollout may arm staging computation for blocks it expects the resident
// subject to demand, but it cannot confirm its own predictions, mint world
// evidence, alter logical identity or touch causal semantics.  Staged bytes
// leave the window only through actual demand; never-demanded staging is
// discarded as a misprediction and every trajectory stays bit-identical to a
// no-prefetch run.  Staging decisions are device-owned; the host moves bulk
// bytes at most.
inline constexpr std::uint32_t kPredictivePrefetchStagingVersion = 1u;
inline constexpr std::uint32_t kPrefetchStagingSlotCapacity = 4u;

// Availability accounting: a demand answered through the hot staging window
// costs one window probe; a missed demand pays the full cold-fetch paging
// round trip (arm request, bulk transport, device commit).
inline constexpr std::uint32_t kStagedDemandSteps = 1u;
inline constexpr std::uint32_t kColdFetchSteps = 3u;

enum class PrefetchStageRequest : std::uint32_t {
  none = 0u,
  stage = 1u,
};

struct DirectPrefetchStagedSlotV1 {
  recipe::Root256 address;
  std::uint64_t logical_recipe_id;
  std::uint64_t block_bytes;
  std::uint64_t nominated_tick;
  std::uint64_t last_touch_tick;
  DirectSpeculativeProvenance provenance;
  std::uint32_t demand_hits;
  std::uint32_t confirmed;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectPrefetchStagedSlotV1> &&
              std::is_standard_layout_v<DirectPrefetchStagedSlotV1> &&
              std::has_unique_object_representations_v<DirectPrefetchStagedSlotV1>);

struct DirectPrefetchBlockTableEntryV1 {
  std::uint64_t logical_recipe_id;
  recipe::Root256 address;
  std::uint32_t present;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectPrefetchBlockTableEntryV1> &&
              std::is_standard_layout_v<DirectPrefetchBlockTableEntryV1>);

struct DirectPrefetchStagingStateV1 {
  std::uint64_t staging_capacity_bytes;
  std::uint64_t staged_bytes;
  std::uint64_t rollout_root;
  std::uint32_t nominations;
  std::uint32_t mispredictions;
  std::uint32_t staged_demand_serves;
  std::uint32_t cold_fetches;
  std::uint32_t evictions;
  std::uint32_t slot_count;
  DirectPrefetchStagedSlotV1 slots[kPrefetchStagingSlotCapacity];
  PrefetchStageRequest request;
  recipe::Root256 requested_address;
  std::uint32_t version;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectPrefetchStagingStateV1> &&
              std::is_standard_layout_v<DirectPrefetchStagingStateV1>);

__host__ __device__ inline bool prefetch_root_zero(const recipe::Root256& root) {
  for (std::uint32_t i = 0u; i < 8u; ++i)
    if (root.word[i] != 0u)
      return false;
  return true;
}
__host__ __device__ inline bool same_prefetch_root(const recipe::Root256& left,
                                                   const recipe::Root256& right) {
  for (std::uint32_t i = 0u; i < 8u; ++i)
    if (left.word[i] != right.word[i])
      return false;
  return true;
}
__host__ __device__ inline ResidentRecipeCell probe_cell(std::uint64_t logical_recipe_id) {
  ResidentRecipeCell cell{};
  cell.logical_recipe_id = logical_recipe_id;
  return cell;
}

__host__ __device__ inline bool remove_prefetch_slot(
    DirectPrefetchStagingStateV1* state, std::uint32_t index) {
  if (state == nullptr || index >= state->slot_count)
    return false;
  state->staged_bytes -= state->slots[index].block_bytes;
  for (std::uint32_t i = index; i + 1u < state->slot_count; ++i)
    state->slots[i] = state->slots[i + 1u];
  --state->slot_count;
  state->slots[state->slot_count] = DirectPrefetchStagedSlotV1{};
  return true;
}

// Every staged slot stays speculative for its whole residency, whether or not
// demand later confirms it; confirmation is a performance fact, never an
// evidence upgrade.  A state carrying any other tag is not a lawful staging
// state, so host-side forgery of verified observation is rejected wholesale.
__host__ __device__ inline bool prefetch_staging_state_valid(
    const DirectPrefetchStagingStateV1& state) {
  if (state.version != kPredictivePrefetchStagingVersion ||
      state.slot_count > kPrefetchStagingSlotCapacity ||
      state.staged_bytes > state.staging_capacity_bytes)
    return false;
  if (state.request != PrefetchStageRequest::none &&
      state.request != PrefetchStageRequest::stage)
    return false;
  if (state.request == PrefetchStageRequest::stage &&
      prefetch_root_zero(state.requested_address))
    return false;
  if (state.request == PrefetchStageRequest::none &&
      !prefetch_root_zero(state.requested_address))
    return false;
  std::uint64_t bytes = 0u;
  for (std::uint32_t i = 0u; i < state.slot_count; ++i) {
    const DirectPrefetchStagedSlotV1& slot = state.slots[i];
    if (prefetch_root_zero(slot.address) || slot.block_bytes == 0u ||
        slot.provenance != DirectSpeculativeProvenance::endogenous_simulation)
      return false;
    bytes += slot.block_bytes;
  }
  return bytes == state.staged_bytes;
}

__host__ __device__ inline bool initialize_prefetch_staging_state(
    std::uint64_t capacity_bytes, std::uint64_t rollout_root,
    DirectPrefetchStagingStateV1* state) {
  if (state == nullptr || capacity_bytes == 0u)
    return false;
  *state = {};
  state->staging_capacity_bytes = capacity_bytes;
  state->rollout_root = rollout_root;
  state->version = kPredictivePrefetchStagingVersion;
  return true;
}

// Prospective shadow rollout over the resident touch trace: deterministically
// projects the next block index from the recent touch tail under a candidate
// domain.  Purely prospective -- it reads resident state and yields a
// nomination; it writes no history and settles nothing.
__host__ __device__ inline std::uint64_t prefetch_rollout_mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__host__ __device__ inline std::uint32_t project_next_block_index(
    const std::uint32_t* recent_indices, std::uint32_t recent_count,
    const std::uint32_t* candidate_domain, std::uint32_t domain_count,
    std::uint64_t rollout_root) {
  if (recent_indices == nullptr || recent_count == 0u ||
      candidate_domain == nullptr || domain_count == 0u)
    return 0u;
  std::uint64_t fold = rollout_root;
  for (std::uint32_t i = 0u; i < recent_count; ++i)
    fold = prefetch_rollout_mix(fold ^ (static_cast<std::uint64_t>(recent_indices[i]) +
                                         0x165667b19e3779f9ull));
  return candidate_domain[(fold >> 33) % domain_count];
}

__host__ __device__ inline bool resolve_prefetch_block(
    const DirectPrefetchBlockTableEntryV1* table, std::uint32_t table_count,
    std::uint64_t logical_recipe_id, recipe::Root256* address) {
  if (table == nullptr || address == nullptr || logical_recipe_id == 0u)
    return false;
  for (std::uint32_t i = 0u; i < table_count; ++i)
    if (table[i].present != 0u && table[i].logical_recipe_id == logical_recipe_id) {
      *address = table[i].address;
      return true;
    }
  return false;
}

// Arm one staging request from a rollout nomination.  The nomination names a
// block identity resolved through the device-owned block table; unknown
// identities refuse because nothing outside resident state may be staged.
__host__ __device__ inline bool nominate_prefetch_stage(
    const DirectPrefetchBlockTableEntryV1* table, std::uint32_t table_count,
    std::uint64_t logical_recipe_id, DirectPrefetchStagingStateV1* state) {
  recipe::Root256 address{};
  if (state == nullptr || !prefetch_staging_state_valid(*state) ||
      state->request != PrefetchStageRequest::none ||
      !resolve_prefetch_block(table, table_count, logical_recipe_id, &address))
    return false;
  state->requested_address = address;
  state->request = PrefetchStageRequest::stage;
  ++state->nominations;
  return true;
}

// Device commit arm: the staged block enters the hot window only with a
// valid device-armed request, a content-address matched block and room inside
// the bounded staging budget.  Provenance is stamped speculative here and can
// never be upgraded through staging.
__host__ __device__ inline bool commit_prefetch_stage(
    const DirectDormantRecipePageV1& page, std::uint64_t tick,
    DirectPrefetchStagingStateV1* state) {
  if (state == nullptr || !prefetch_staging_state_valid(*state) ||
      state->request != PrefetchStageRequest::stage ||
      !dormant_recipe_page_valid(page) ||
      !same_prefetch_root(dormant_recipe_page_address(page),
                          state->requested_address) ||
      state->slot_count == kPrefetchStagingSlotCapacity ||
      state->staged_bytes + sizeof(page) > state->staging_capacity_bytes)
    return false;
  DirectPrefetchStagedSlotV1 slot{};
  slot.address = state->requested_address;
  slot.logical_recipe_id = page.logical_recipe_id;
  slot.block_bytes = sizeof(page);
  slot.nominated_tick = tick;
  slot.last_touch_tick = tick;
  slot.provenance = DirectSpeculativeProvenance::endogenous_simulation;
  state->slots[state->slot_count++] = slot;
  state->staged_bytes += sizeof(page);
  state->request = PrefetchStageRequest::none;
  state->requested_address = recipe::Root256{};
  return true;
}

__host__ __device__ inline std::int32_t prefetch_slot_for(
    const DirectPrefetchStagingStateV1& state, const recipe::Root256& address) {
  for (std::uint32_t i = 0u; i < state.slot_count; ++i)
    if (same_prefetch_root(state.slots[i].address, address))
      return static_cast<std::int32_t>(i);
  return -1;
}

// Actual demand arrives for one block.  A staged hit is answered from the hot
// window in one step and marks the slot demand-confirmed; a miss falls back to
// the ordinary cold-fetch path.  Either way the demand sees exactly the block
// it asked for -- staging never substitutes content.
__host__ __device__ inline bool prefetch_demand_arrive(
    const recipe::Root256& address, std::uint64_t tick,
    DirectPrefetchStagingStateV1* state, bool* served_from_staging) {
  if (state == nullptr || served_from_staging == nullptr ||
      prefetch_root_zero(address) || !prefetch_staging_state_valid(*state))
    return false;
  const std::int32_t index = prefetch_slot_for(*state, address);
  if (index < 0) {
    ++state->cold_fetches;
    *served_from_staging = false;
    return true;
  }
  DirectPrefetchStagedSlotV1& slot = state->slots[index];
  ++slot.demand_hits;
  slot.last_touch_tick = tick;
  slot.confirmed = 1u;
  ++state->staged_demand_serves;
  *served_from_staging = true;
  return true;
}

// Graduate a demand-confirmed block out of the speculative window into the
// real paging path, handing back its block identity.  Only a slot that actual
// demand touched may be consumed: prediction moves bytes, never binds them.
__host__ __device__ inline bool consume_staged_block(
    const recipe::Root256& address, std::uint64_t* logical_recipe_id,
    DirectPrefetchStagingStateV1* state) {
  if (state == nullptr || logical_recipe_id == nullptr ||
      prefetch_root_zero(address) || !prefetch_staging_state_valid(*state))
    return false;
  const std::int32_t index = prefetch_slot_for(*state, address);
  if (index < 0 || state->slots[index].demand_hits == 0u)
    return false;
  *logical_recipe_id = state->slots[index].logical_recipe_id;
  return remove_prefetch_slot(state, static_cast<std::uint32_t>(index));
}

// Discard a never-demanded staged block.  Mispredictions cost nothing but the
// window bookkeeping: no trajectory byte, history record or identity changes.
__host__ __device__ inline bool discard_unused_staged(
    const recipe::Root256& address, DirectPrefetchStagingStateV1* state) {
  if (state == nullptr || prefetch_root_zero(address) ||
      !prefetch_staging_state_valid(*state))
    return false;
  const std::int32_t index = prefetch_slot_for(*state, address);
  if (index < 0 || state->slots[index].demand_hits != 0u)
    return false;
  ++state->mispredictions;
  return remove_prefetch_slot(state, static_cast<std::uint32_t>(index));
}

// Budget pressure evicts the least-recently-used staged block, but a block
// whose recipe still has a live occurrence is pinned: staging capacity never
// disturbs live bindings.  An all-pinned window refuses rather than evict.
__host__ __device__ inline bool evict_prefetch_lru(
    const direct_adult_core::ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count, DirectPrefetchStagingStateV1* state,
    recipe::Root256* evicted_address) {
  if (state == nullptr || evicted_address == nullptr ||
      !prefetch_staging_state_valid(*state) || state->slot_count == 0u)
    return false;
  std::int32_t victim = -1;
  for (std::uint32_t i = 0u; i < state->slot_count; ++i) {
    if (resident_live_occurrence_count(occurrences, occurrence_count,
                                       probe_cell(state->slots[i].logical_recipe_id)) != 0u)
      continue;
    if (victim < 0 ||
        state->slots[i].last_touch_tick < state->slots[victim].last_touch_tick)
      victim = static_cast<std::int32_t>(i);
  }
  if (victim < 0)
    return false;
  *evicted_address = state->slots[victim].address;
  ++state->evictions;
  return remove_prefetch_slot(state, static_cast<std::uint32_t>(victim));
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_PREDICTIVE_PREFETCH_CUH
