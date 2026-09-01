#include "hardware_native/direct_network_basin_probe.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>
#include <vector>

namespace substrate::direct_network::basin_probe {
namespace {

using substrate::direct_adult_core::ActivityEvent;
using substrate::direct_adult_core::CausalOrigin;
using substrate::direct_adult_core::create_direct_adult_runtime;
using substrate::direct_adult_core::destroy_direct_adult_runtime;
using substrate::direct_adult_core::DirectAdultRuntime;
using substrate::direct_adult_core::inject_sensory_event;
using substrate::direct_adult_core::step_direct_adult_fixed_morphology_epochs;

// ---------------------------------------------------------------------------
// Device clone.
//
// Every pool of a grown brain lives inside its one arena -- t0's matter-closure
// requirement asserts exactly that, and it is what makes this function a
// pointer-offset walk rather than a pool-by-pool deep copy that would have to
// be revised every time the brain gains a field. A pointer found OUTSIDE the
// arena is counted and left alone: the clone is then not faithful, and the
// result says so instead of silently sharing that pool with the original.
// ---------------------------------------------------------------------------

template <typename T>
void repair(T*& pointer, const void* old_arena, void* new_arena, std::uint64_t bytes,
            std::uint32_t* repaired, std::uint32_t* outside) {
  if (pointer == nullptr) return;
  const auto base = reinterpret_cast<std::uintptr_t>(old_arena);
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address < base || address >= base + bytes) {
    ++*outside;
    return;
  }
  pointer = reinterpret_cast<T*>(static_cast<char*>(new_arena) + (address - base));
  ++*repaired;
}

// `resource_ecology` is the #1178 ledger and is its own cudaMalloc beside the
// arena -- measured by this very function on its first run, when it reported a
// pool it could not reach by offset. It has a fixed size, so the clone
// deep-copies it rather than aliasing the original: an aliased ledger would let
// the probe's eight cloned runs charge matter against the organism being
// certified.
bool deep_copy_ecology(DirectBrain* out, const DirectBrain& source) {
  using substrate::direct_adult::DirectResourceEcologyState;
  const std::size_t bytes = sizeof(DirectResourceEcologyState);
  void* copy = nullptr;
  if (cudaMalloc(&copy, bytes) != cudaSuccess) return false;
  if (cudaMemcpy(copy, source.resource_ecology, bytes, cudaMemcpyDeviceToDevice) !=
      cudaSuccess) {
    cudaFree(copy);
    return false;
  }
  out->resource_ecology = static_cast<DirectResourceEcologyState*>(copy);
  return true;
}

bool clone_brain_device(const DirectBrain& source, DirectBrain* out,
                        std::uint32_t* repaired, std::uint32_t* outside,
                        std::uint32_t* deep_copied) {
  if (source.arena == nullptr || source.arena_bytes == 0u) return false;
  *out = source;
  void* arena = nullptr;
  if (cudaMalloc(&arena, source.arena_bytes) != cudaSuccess) return false;
  if (cudaMemcpy(arena, source.arena, source.arena_bytes, cudaMemcpyDeviceToDevice) !=
      cudaSuccess) {
    cudaFree(arena);
    return false;
  }
  out->arena = arena;
  const void* old_arena = source.arena;
  const std::uint64_t bytes = source.arena_bytes;
  repair(out->nodes, old_arena, arena, bytes, repaired, outside);
  repair(out->routes, old_arena, arena, bytes, repaired, outside);
  repair(out->route_incarnations, old_arena, arena, bytes, repaired, outside);
  repair(out->route_opportunity_incarnations, old_arena, arena, bytes, repaired, outside);
  repair(out->route_delay_law_indices, old_arena, arena, bytes, repaired, outside);
  repair(out->route_mature_delays, old_arena, arena, bytes, repaired, outside);
  repair(out->route_delay_law_incarnations, old_arena, arena, bytes, repaired, outside);
  repair(out->dense_blocks, old_arena, arena, bytes, repaired, outside);
  repair(out->dense_weight_fp16_bits, old_arena, arena, bytes, repaired, outside);
  repair(out->boundary_ports, old_arena, arena, bytes, repaired, outside);
  repair(out->territory_ancestry, old_arena, arena, bytes, repaired, outside);
  repair(out->resident_fields, old_arena, arena, bytes, repaired, outside);
  repair(out->resident_field_ranges, old_arena, arena, bytes, repaired, outside);
  repair(out->resident_field_indices, old_arena, arena, bytes, repaired, outside);
  repair(out->resident_rules, old_arena, arena, bytes, repaired, outside);
  repair(out->resident_tract_delay_laws, old_arena, arena, bytes, repaired, outside);
  repair(out->recipe_cells, old_arena, arena, bytes, repaired, outside);
  repair(out->recipe_edges, old_arena, arena, bytes, repaired, outside);
  repair(out->recipe_ranges, old_arena, arena, bytes, repaired, outside);
  repair(out->recipe_indices, old_arena, arena, bytes, repaired, outside);
  repair(out->development, old_arena, arena, bytes, repaired, outside);
  repair(out->construction_fronts, old_arena, arena, bytes, repaired, outside);
  repair(out->construction_front_count, old_arena, arena, bytes, repaired, outside);
  repair(out->construction_front_generation_by_node, old_arena, arena, bytes,
         repaired, outside);
  repair(out->postbirth_derivations, old_arena, arena, bytes, repaired, outside);
  repair(out->postbirth_constructor, old_arena, arena, bytes, repaired, outside);
  repair(out->retention_bank, old_arena, arena, bytes, repaired, outside);

  if (out->resource_ecology != nullptr) {
    const auto base = reinterpret_cast<std::uintptr_t>(old_arena);
    const auto address = reinterpret_cast<std::uintptr_t>(out->resource_ecology);
    if (address >= base && address < base + bytes) {
      repair(out->resource_ecology, old_arena, arena, bytes, repaired, outside);
    } else if (deep_copy_ecology(out, source)) {
      ++*deep_copied;
    } else {
      ++*outside;
    }
  }
  return true;
}

void destroy_clone(DirectBrain* clone, const DirectBrain& source) {
  if (clone == nullptr) return;
  if (clone->resource_ecology != nullptr && clone->resource_ecology != source.resource_ecology) {
    const auto base = reinterpret_cast<std::uintptr_t>(clone->arena);
    const auto address = reinterpret_cast<std::uintptr_t>(clone->resource_ecology);
    if (clone->arena == nullptr || address < base || address >= base + clone->arena_bytes)
      cudaFree(clone->resource_ecology);
  }
  if (clone->arena != nullptr) cudaFree(clone->arena);
  clone->arena = nullptr;
  clone->resource_ecology = nullptr;
}

// The basin probe is an observer-side intervention on an isolated clone, not a
// physical contact with the certified organism. Since #1452, ordinary ingress
// correctly refuses any event that is not attached to a born sensor port. The
// old probe defeated its own instrument by deleting every clone port and then
// asking inject_sensory_event() to accept a node-addressed "external_contact".
//
// Preserve the hardened ingress law instead of reopening that backdoor: one
// temporary sensor port is allocated beside the clone arena and names only the
// already-derived seed node plus an opaque probe channel. create_direct_adult_runtime
// caches it through the same boundary-port path as a born body; the certified
// source brain, its body root, and its arena are untouched. The port is freed
// before the clone is destroyed.
bool install_probe_sensor_port(DirectBrain* clone, std::uint32_t node,
                               std::uint32_t channel,
                               DirectBoundaryPort** owned_port) {
  if (clone == nullptr || owned_port == nullptr || node >= clone->node_count)
    return false;
  *owned_port = nullptr;
  DirectBoundaryPort port{};
  port.node = node;
  port.channel = channel;
  port.role_mask = static_cast<std::uint32_t>(BoundaryRole::sensor);
  port.physical_route = 0xB0510000u ^ channel;
  port.parent_route = 0u;
  DirectBoundaryPort* device_port = nullptr;
  if (cudaMalloc(&device_port, sizeof(port)) != cudaSuccess) return false;
  if (cudaMemcpy(device_port, &port, sizeof(port), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(device_port);
    return false;
  }
  clone->boundary_ports = device_port;
  clone->boundary_port_count = 1u;
  *owned_port = device_port;
  return true;
}

void release_probe_sensor_port(DirectBrain* clone,
                               DirectBoundaryPort* owned_port) {
  if (owned_port != nullptr) cudaFree(owned_port);
  if (clone != nullptr) {
    clone->boundary_ports = nullptr;
    clone->boundary_port_count = 0u;
  }
}

// ---------------------------------------------------------------------------
// Post-hoc seed selection, on device, from grown state.
// ---------------------------------------------------------------------------

// kAnyTerritory is defined publicly in the header (BasinProbeConfig::
// required_territory needs it as a default value); this anonymous namespace
// resolves the unqualified name to that outer declaration.
// `territory_index` is a uint16, so this is the exact size of its value space.
// The census is a presence sweep over it rather than a max-then-allocate,
// because the two cost the same at 256 KiB and the sweep cannot be fooled by a
// sparse or non-contiguous territory numbering.
constexpr std::uint32_t kTerritorySlots = 65536u;

// Score packs the grown quantity above the node index so a single atomicMax
// picks the highest-scoring node and breaks ties by LOWEST index -- otherwise
// the seed set would depend on block scheduling and the whole probe would stop
// being reproducible across replicas.
//
// `require_territory` narrows the CANDIDATE SET and touches nothing else. The
// packing, the ranking quantity and the tie-break are identical in every mode,
// so a stratified selection is the same measurement asked of a subset of the
// tissue -- still post hoc, still derived, still order-independent (atomicMax
// over a fixed candidate set is commutative and associative, so the winner
// cannot depend on which block got there first).
__global__ void score_nodes_kernel(const DirectNode* __restrict__ nodes,
                                   std::uint32_t node_count,
                                   const std::uint32_t* __restrict__ taken,
                                   bool authored_index_seeds,
                                   std::uint32_t require_territory,
                                   unsigned long long* __restrict__ best) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  if (taken[index] != 0u) return;
  const DirectNode node = nodes[index];
  if (require_territory != kAnyTerritory &&
      static_cast<std::uint32_t>(node.territory_index) != require_territory)
    return;
  // The sham path deliberately ignores every grown quantity and ranks by
  // position alone. It exists so the contract can show the two seed sets
  // differ; if they did not, "post hoc from mature tissue" would be describing
  // a selection that position alone already reproduces.
  const std::uint64_t score =
      authored_index_seeds
          ? static_cast<std::uint64_t>(node_count - index)
          : (static_cast<std::uint64_t>(node.maturation_q16) * 64ull +
             static_cast<std::uint64_t>(node.active_route_count));
  const unsigned long long key =
      (score << 32) | static_cast<unsigned long long>(0xffffffffu - index);
  atomicMax(best, key);
}

// Which territories the TISSUE has, asked of the tissue. A plain store: every
// writer writes the same value, so no atomic is needed and none is used.
__global__ void census_territories_kernel(const DirectNode* __restrict__ nodes,
                                          std::uint32_t node_count,
                                          std::uint32_t* __restrict__ present) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  present[nodes[index].territory_index] = 1u;
}

__global__ void census_inhibitory_kernel(const DirectNode* __restrict__ nodes,
                                         const DirectRoute* __restrict__ routes,
                                         std::uint32_t node_count,
                                         std::uint32_t route_capacity,
                                         std::uint32_t* __restrict__ node_total,
                                         std::uint32_t* __restrict__ route_total,
                                         std::uint32_t* __restrict__ first_inhibitory_index) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  const DirectNode node = nodes[index];
  if ((node.flags & kNodeFlagInhibitory) != 0u) {
    atomicAdd(node_total, 1u);
    // gh #1310: where the inhibitory tissue begins, so the sham can be drawn
    // from the same neighbourhood instead of from index 0. Node indices are
    // territory-ordered (`direct_network_life_function.cu:2529-2532` assigns
    // `plan.node_offset` from a running counter over seeds), so this is also
    // the first territory that grows any.
    atomicMin(first_inhibitory_index, index);
  }
  const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
  if (end > route_capacity) return;
  std::uint32_t inhibitory = 0u;
  for (std::uint32_t r = node.route_offset; r < static_cast<std::uint32_t>(end); ++r)
    if ((routes[r].flags & kRouteFlagInhibitory) != 0u) ++inhibitory;
  if (inhibitory != 0u) atomicAdd(route_total, inhibitory);
}

// Ablate the SIGN (flags) or the THRESHOLD (inhibition_q16) on the clone's grown
// tissue. In sham mode the same operations run on the same COUNT of nodes this
// genome grows non-inhibitory -- identical write count, no inhibition removed.
// If the sham moves the outcome, the outcome is about touching memory rather
// than about inhibition.
//
// gh #1310: the sham used to be `index < sham_target_count`, i.e. the low end
// of the global index range. Node indices are territory-ordered, so that drew
// every sham node from the FIRST territory while the real lesion sits in
// whichever territories actually grew inhibitory -- 23/24/25 for the canonical
// species. A control placed ~23 territories away from the lesion it controls
// for is inert whenever the probed basin does not reach that far, and it is
// then inert BY CONSTRUCTION rather than because touching matter is harmless.
// It is now drawn from `sham_index_base`, the first inhibitory node's index.
__global__ void ablate_inhibition_kernel(DirectNode* __restrict__ nodes,
                                         DirectRoute* __restrict__ routes,
                                         std::uint32_t node_count,
                                         std::uint32_t route_capacity,
                                         std::uint32_t sham_target_count,
                                         std::uint32_t sham_index_base,
                                         bool ablate_sign, bool ablate_threshold, bool sham,
                                         std::uint32_t* __restrict__ touched_nodes,
                                         std::uint32_t* __restrict__ touched_routes,
                                         std::uint32_t* __restrict__ first_touched_node) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  DirectNode& node = nodes[index];
  const bool inhibitory = (node.flags & kNodeFlagInhibitory) != 0u;
  bool selected;
  if (sham) {
    // Rank by counting the non-inhibitory nodes below this one, from the base
    // up. No atomics and no prefix-sum buffer, so the selected SET is a pure
    // function of the grown tissue -- the run-to-run determinism the ablation
    // arms measure is unchanged. ponytail: O(n) per thread over node_count
    // (5568 today, once per probe); swap in a prefix sum if a genome ever
    // makes this the probe's hot loop.
    selected = false;
    if (!inhibitory && index >= sham_index_base) {
      std::uint32_t rank = 0u;
      for (std::uint32_t i = sham_index_base; i < index; ++i)
        if ((nodes[i].flags & kNodeFlagInhibitory) == 0u) ++rank;
      selected = rank < sham_target_count;
    }
  } else {
    selected = inhibitory;
  }
  if (!selected) return;

  atomicAdd(touched_nodes, 1u);
  // Where the lesion actually landed. Without it a sham that failed to relocate
  // and a sham that relocated and turned out harmless read identically -- both
  // just report "moved nothing".
  atomicMin(first_touched_node, index);
  if (ablate_sign) node.flags &= ~kNodeFlagInhibitory;
  if (ablate_threshold) node.inhibition_q16 = 0;

  const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
  if (end > route_capacity) return;
  for (std::uint32_t r = node.route_offset; r < static_cast<std::uint32_t>(end); ++r) {
    atomicAdd(touched_routes, 1u);
    if (ablate_sign) routes[r].flags &= ~kRouteFlagInhibitory;
  }
}

// Peak slow-context value anywhere, so an arm can show its ceiling actually
// binds rather than assuming it.
__global__ void max_slow_context_kernel(const std::int32_t* __restrict__ slow_context,
                                        std::uint32_t node_count, int* __restrict__ peak) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  atomicMax(peak, slow_context[index]);
}

// One breadth-first hop over the ACTIVE route graph. A node in the current
// frontier contributes every active route's target; a target not already reached
// becomes newly reached and joins the next frontier. Level-synchronous, so hop
// count is exactly the executor's tick count.
__global__ void expand_reachable_kernel(const DirectNode* __restrict__ nodes,
                                        const DirectRoute* __restrict__ routes,
                                        std::uint32_t node_count,
                                        std::uint32_t route_capacity,
                                        const std::uint32_t* __restrict__ frontier,
                                        std::uint32_t* __restrict__ reached,
                                        std::uint32_t* __restrict__ next_frontier,
                                        std::uint32_t* __restrict__ added) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  if (((frontier[index >> 5] >> (index & 31u)) & 1u) == 0u) return;

  const DirectNode node = nodes[index];
  const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
  if (end > route_capacity) return;
  for (std::uint32_t r = node.route_offset; r < static_cast<std::uint32_t>(end); ++r) {
    const DirectRoute route = routes[r];
    if ((route.flags & kRouteFlagActive) == 0u) continue;
    const std::uint32_t target = route.target;
    if (target >= node_count) continue;
    const std::uint32_t word = target >> 5;
    const std::uint32_t bit = 1u << (target & 31u);
    if ((atomicOr(&reached[word], bit) & bit) != 0u) continue;  // already reached
    atomicOr(&next_frontier[word], bit);
    atomicAdd(added, 1u);
  }
}

// Break a reachable set down by territory, and count the long tracts leaving it
// AGAINST the active routes leaving the same nodes.
//
// The denominator is counted in this same loop, off the same `route` load, so
// it cannot drift from the numerator: any slice bound, any activity filter, any
// skipped node applies identically to both. A denominator gathered by a second
// pass could disagree with the numerator about which routes were even examined,
// and the ratio would then be a ratio of two different populations.
//
// gh #1243: `kRouteFlagLongTract` here MUST be the direct_network one (1u << 1).
// direct_adult's identically-named constant is 1u << 0, which is this file's
// kRouteFlagActive -- and every grown route carries Active, so the wrong
// constant makes `long_tracts` equal `active_routes` exactly. That is why both
// are reported: the equality is checkable, and the contract checks it.
//
// gh #1300: `inhibitory_routes` is added in the SAME pass for the SAME reason
// -- a denominator gathered by a second pass could disagree with either
// numerator about which routes were examined, and #1293 already found that
// reading a bare count (no denominator at all) reads as a size effect
// wearing a causal explanation's clothes.
__global__ void census_reachable_kernel(const DirectNode* __restrict__ nodes,
                                        const DirectRoute* __restrict__ routes,
                                        std::uint32_t node_count,
                                        std::uint32_t route_capacity,
                                        const std::uint32_t* __restrict__ reached,
                                        std::uint32_t* __restrict__ per_territory,
                                        std::uint32_t* __restrict__ long_tracts,
                                        std::uint32_t* __restrict__ active_routes,
                                        std::uint32_t* __restrict__ inhibitory_routes) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  if (((reached[index >> 5] >> (index & 31u)) & 1u) == 0u) return;
  const DirectNode node = nodes[index];
  if (node.territory_index < BasinProbeResult::kMaxCensusTerritories)
    atomicAdd(&per_territory[node.territory_index], 1u);
  const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
  if (end > route_capacity) return;
  std::uint32_t local_active = 0u;
  std::uint32_t local_long = 0u;
  std::uint32_t local_inhibitory = 0u;
  for (std::uint32_t r = node.route_offset; r < static_cast<std::uint32_t>(end); ++r) {
    const DirectRoute route = routes[r];
    if ((route.flags & kRouteFlagActive) == 0u) continue;
    ++local_active;
    if ((route.flags & kRouteFlagLongTract) != 0u) ++local_long;
    if ((route.flags & kRouteFlagInhibitory) != 0u) ++local_inhibitory;
  }
  if (local_active != 0u) atomicAdd(active_routes, local_active);
  if (local_long != 0u) atomicAdd(long_tracts, local_long);
  if (local_inhibitory != 0u) atomicAdd(inhibitory_routes, local_inhibitory);
}

// The tree-wide baseline: every active route in the organism, and how many of
// them are long tracts or inhibitory. A per-seed exposure of 74 is unreadable
// without it -- against 240 tree-wide it is most of the cross-territory
// wiring, against 24000 it is a fringe. The walk is per NODE SLICE rather than
// over the flat route array so it counts exactly what the reachability
// expansion can traverse: a route sitting outside every node's
// [route_offset, route_offset+capacity) window is unreachable by the executor
// and must not inflate the denominator.
__global__ void census_route_totals_kernel(const DirectNode* __restrict__ nodes,
                                           const DirectRoute* __restrict__ routes,
                                           std::uint32_t node_count,
                                           std::uint32_t route_capacity,
                                           std::uint32_t* __restrict__ active_routes,
                                           std::uint32_t* __restrict__ long_tracts,
                                           std::uint32_t* __restrict__ inhibitory_routes) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  const DirectNode node = nodes[index];
  const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
  if (end > route_capacity) return;
  std::uint32_t local_active = 0u;
  std::uint32_t local_long = 0u;
  std::uint32_t local_inhibitory = 0u;
  for (std::uint32_t r = node.route_offset; r < static_cast<std::uint32_t>(end); ++r) {
    const DirectRoute route = routes[r];
    if ((route.flags & kRouteFlagActive) == 0u) continue;
    ++local_active;
    if ((route.flags & kRouteFlagLongTract) != 0u) ++local_long;
    if ((route.flags & kRouteFlagInhibitory) != 0u) ++local_inhibitory;
  }
  if (local_active != 0u) atomicAdd(active_routes, local_active);
  if (local_long != 0u) atomicAdd(long_tracts, local_long);
  if (local_inhibitory != 0u) atomicAdd(inhibitory_routes, local_inhibitory);
}

// Territory sizes, so a reachable count can be read as a FRACTION of the
// territories it spans rather than as a bare number.
__global__ void census_territory_sizes_kernel(const DirectNode* __restrict__ nodes,
                                              std::uint32_t node_count,
                                              std::uint32_t* __restrict__ sizes) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  const std::uint16_t territory = nodes[index].territory_index;
  if (territory < BasinProbeResult::kMaxCensusTerritories) atomicAdd(&sizes[territory], 1u);
}

__global__ void seed_bitmap_kernel(std::uint32_t* bitmap, std::uint32_t index) {
  bitmap[index >> 5] |= 1u << (index & 31u);
}

__global__ void mark_taken_kernel(std::uint32_t* taken, std::uint32_t index) {
  taken[index] = 1u;
}

// ---------------------------------------------------------------------------
// Population capture.
// ---------------------------------------------------------------------------

__global__ void capture_population_kernel(const DirectNode* __restrict__ nodes,
                                          std::uint32_t node_count,
                                          std::int32_t threshold_q16,
                                          std::uint32_t* __restrict__ bitmap,
                                          std::uint32_t* __restrict__ count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  if (nodes[index].activation_q16 <= threshold_q16) return;
  atomicOr(&bitmap[index >> 5], 1u << (index & 31u));
  atomicAdd(count, 1u);
}

// Exact order-independent fold. Multiplying each node's contribution by an
// index-derived odd constant before adding makes the sum position-sensitive, so
// two states with the same multiset of activations but different assignments do
// not collide.
__global__ void digest_state_kernel(const DirectNode* __restrict__ nodes,
                                    std::uint32_t node_count,
                                    unsigned long long* __restrict__ digest,
                                    int* __restrict__ census) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  const DirectNode node = nodes[index];
  const unsigned long long weight = (static_cast<unsigned long long>(index) << 1) | 1ull;
  // MEASURED: an earlier version folded only activation and attractor support
  // and reported zero state change for every seed. That was the DIGEST being
  // too narrow, not the injection being inert -- the ingest path also stamps
  // last_actual_tick, and when a node is already saturated at kQ16One, setting
  // it to kQ16One is invisible to an activation-only fold. Every node field the
  // reaction law writes is folded here, so "no state change" means no state
  // change.
  const unsigned long long value =
      (static_cast<unsigned long long>(static_cast<std::uint32_t>(node.activation_q16)) * 0x9E3779B1ull) +
      (static_cast<unsigned long long>(static_cast<std::uint32_t>(node.attractor_support_q16)) * 0x85EBCA77ull) +
      (static_cast<unsigned long long>(static_cast<std::uint32_t>(node.activity_ema_q16)) * 0xC2B2AE3Dull) +
      (static_cast<unsigned long long>(static_cast<std::uint32_t>(node.credit_ema_q16)) * 0x27D4EB2Full) +
      (static_cast<unsigned long long>(static_cast<std::uint32_t>(node.inhibition_q16)) * 0x165667B1ull) +
      (static_cast<unsigned long long>(node.competition_strength_code_q16) * 0x517CC1B7ull) +
      (static_cast<unsigned long long>(node.last_actual_tick) * 0xD6E8FEB1ull) +
      (static_cast<unsigned long long>(node.last_endogenous_tick) * 0xCC9E2D51ull) +
      (static_cast<unsigned long long>(node.refractory_until) * 0x1B873593ull);
  atomicAdd(digest, value * weight);
  atomicMin(&census[0], node.activation_q16);
  atomicMax(&census[1], node.activation_q16);
  atomicMin(&census[2], node.attractor_support_q16);
  atomicMax(&census[3], node.attractor_support_q16);
}

std::uint32_t popcount32(std::uint32_t value) {
  std::uint32_t count = 0u;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
}

// active(i) && !active(j) && active(k) for some i < j < k. Persistence alone
// cannot satisfy this, which is the point.
bool mask_shows_return(std::uint32_t mask, std::uint32_t horizon_count) {
  for (std::uint32_t i = 0u; i + 2u < horizon_count; ++i) {
    if (((mask >> i) & 1u) == 0u) continue;
    for (std::uint32_t j = i + 1u; j + 1u < horizon_count; ++j) {
      if (((mask >> j) & 1u) != 0u) continue;
      for (std::uint32_t k = j + 1u; k < horizon_count; ++k)
        if (((mask >> k) & 1u) != 0u) return true;
    }
  }
  return false;
}

}  // namespace

BasinProbeResult probe_basins(const DirectBrain& brain, const BasinProbeConfig& config,
                              const AdultExecutionConfig& execution) {
  // The replay contract compares the complete observer result, including the
  // padding between its fixed-width fields.  Make that representation
  // canonical instead of relying on aggregate-initialization padding.
  static_assert(std::is_trivially_copyable_v<BasinProbeResult>);
  BasinProbeResult result;
  std::memset(&result, 0, sizeof(result));
  // The sentinel, not zero: a census that never runs must not read as "the
  // inhibitory tissue starts at node 0".
  result.first_inhibitory_node = kNoInhibitoryTissue;
  result.lesion_first_node = kNoInhibitoryTissue;
  result.activation_threshold_q16 = config.activation_threshold_q16;
  result.required_territory = config.required_territory;
  result.stratification = static_cast<std::uint32_t>(config.stratification);
  const std::uint32_t seed_count =
      config.seed_count < kMaxProbeSeeds ? config.seed_count : kMaxProbeSeeds;
  const std::uint32_t horizon_count =
      config.horizon_count < kMaxHorizons ? config.horizon_count : kMaxHorizons;
  if (brain.node_count == 0u || seed_count == 0u || horizon_count == 0u) return result;
  result.seed_count = seed_count;
  result.horizon_count = horizon_count;
  for (std::uint32_t h = 0u; h < horizon_count; ++h)
    result.horizon_ticks[h] = config.horizon_ticks[h];

  const std::uint32_t contacts = config.contacts_per_seed == 0u ? 1u : config.contacts_per_seed;
  const std::uint32_t block = config.block_size == 0u ? 256u : config.block_size;
  const std::uint32_t grid = (brain.node_count + block - 1u) / block;
  const std::uint32_t bitmap_words = (brain.node_count + 31u) / 32u;

  // ---- territories, READ FROM GROWN TISSUE --------------------------------
  //
  // Not from `brain.territory_count` and not from index arithmetic. Territory
  // is stamped per node by the genesis seed, so the header could disagree with
  // the tissue and index arithmetic would invent a partition the Life Function
  // never made. Both are recorded; only the tissue is used.
  std::vector<std::uint32_t> territories;
  {
    std::uint32_t* present = nullptr;
    if (cudaMalloc(&present, sizeof(std::uint32_t) * kTerritorySlots) != cudaSuccess)
      return result;
    cudaMemset(present, 0, sizeof(std::uint32_t) * kTerritorySlots);
    census_territories_kernel<<<grid, block>>>(brain.nodes, brain.node_count, present);
    std::vector<std::uint32_t> host_present(kTerritorySlots, 0u);
    const bool census_ok =
        cudaGetLastError() == cudaSuccess &&
        cudaMemcpy(host_present.data(), present, sizeof(std::uint32_t) * kTerritorySlots,
                   cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(present);
    if (!census_ok) return result;
    for (std::uint32_t t = 0u; t < kTerritorySlots; ++t)
      if (host_present[t] != 0u) territories.push_back(t);
  }
  result.observed_territory_count = static_cast<std::uint32_t>(territories.size());
  result.declared_territory_count = brain.territory_count;
  result.stratification = static_cast<std::uint32_t>(config.stratification);

  // ---- seeds, derived once from the ORIGINAL grown state ------------------
  std::uint32_t* taken = nullptr;
  unsigned long long* best = nullptr;
  if (cudaMalloc(&taken, sizeof(std::uint32_t) * brain.node_count) != cudaSuccess) return result;
  if (cudaMalloc(&best, sizeof(unsigned long long)) != cudaSuccess) {
    cudaFree(taken);
    return result;
  }
  cudaMemset(taken, 0, sizeof(std::uint32_t) * brain.node_count);

  bool ok = true;
  for (std::uint32_t s = 0u; s < seed_count && ok; ++s) {
    // Which territory rank `s` draws from. Ascending observed order, so the
    // assignment is a function of the tissue and the rank alone -- no
    // randomness, no scheduling, nothing authored.
    std::uint32_t require_territory = kAnyTerritory;
    if (config.required_territory != kAnyTerritory) {
      // gh #1359: a caller-named territory overrides stratification entirely
      // for every rank -- the two are mutually exclusive per run, not
      // layered. This is how a seed reaches a family too rare for any
      // stratification mode to hit by arithmetic (NET09's three territories
      // out of 42, past round_robin's reach at ordinary seed counts).
      require_territory = config.required_territory;
    } else if (!territories.empty()) {
      const std::size_t count = territories.size();
      if (config.stratification == SeedStratification::per_territory_then_global) {
        if (static_cast<std::size_t>(s) < count) require_territory = territories[s];
      } else if (config.stratification == SeedStratification::round_robin) {
        require_territory = territories[static_cast<std::size_t>(s) % count];
      } else if (config.stratification == SeedStratification::paired_within_territory) {
        // Two consecutive ranks share a territory. `mark_taken_kernel` has
        // already removed the first of the pair from the candidate set, so the
        // second is the next-highest-scoring node of the SAME territory --
        // still the same ranking quantity, still the same atomicMax, still no
        // authored index.
        require_territory = territories[(static_cast<std::size_t>(s) / 2u) % count];
      }
    }

    unsigned long long key = 0ull;
    for (int attempt = 0; attempt < 2; ++attempt) {
      const std::uint32_t restrict_to = attempt == 0 ? require_territory : kAnyTerritory;
      const unsigned long long zero = 0ull;
      ok = cudaMemcpy(best, &zero, sizeof(zero), cudaMemcpyHostToDevice) == cudaSuccess;
      if (!ok) break;
      score_nodes_kernel<<<grid, block>>>(brain.nodes, brain.node_count, taken,
                                          config.authored_index_seeds, restrict_to, best);
      ok = cudaGetLastError() == cudaSuccess &&
           cudaMemcpy(&key, best, sizeof(key), cudaMemcpyDeviceToHost) == cudaSuccess;
      if (!ok) break;
      // key == 0 is the exact sentinel for "no candidate at all": every key
      // carries (0xffffffff - index) in its low word, which is non-zero for
      // every index the brain can hold. A territory whose nodes are all already
      // taken therefore falls back to the whole organism instead of failing the
      // probe -- a seed_count larger than a small territory must not turn into
      // a probe that did not run.
      if (key != 0ull || restrict_to == kAnyTerritory) break;
    }
    if (!ok) break;
    const std::uint32_t node = 0xffffffffu - static_cast<std::uint32_t>(key & 0xffffffffull);
    if (node >= brain.node_count) {
      ok = false;
      break;
    }
    result.seed_node[s] = node;
    mark_taken_kernel<<<1, 1>>>(taken, node);
    ok = cudaGetLastError() == cudaSuccess;
    DirectNode host_node{};
    if (ok)
      ok = cudaMemcpy(&host_node, brain.nodes + node, sizeof(DirectNode),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
    result.seed_maturation_q16[s] = host_node.maturation_q16;
    result.seed_territory[s] = host_node.territory_index;
  }
  cudaFree(taken);
  cudaFree(best);
  if (!ok) return result;

  // How well the chosen seeds actually cover the tissue. Reported for EVERY
  // mode including the unstratified one, because it is the number that says
  // whether the pair census below could have tested separation at all.
  for (std::uint32_t s = 0u; s < seed_count; ++s) {
    bool seen = false;
    std::uint32_t here = 0u;
    for (std::uint32_t t = 0u; t < seed_count; ++t) {
      if (result.seed_territory[t] != result.seed_territory[s]) continue;
      ++here;
      if (t < s) seen = true;
    }
    if (!seen) ++result.seed_territory_span;
    if (here > result.max_seeds_in_one_territory) result.max_seeds_in_one_territory = here;
  }

  // ---- clones, driven through the organism's own executor -----------------
  std::vector<std::uint32_t> final_population(
      static_cast<std::size_t>(seed_count) * bitmap_words, 0u);
  std::vector<std::uint32_t> ever_active_bits(
      static_cast<std::size_t>(seed_count) * bitmap_words, 0u);
  std::vector<std::uint32_t> baseline_bits(
      static_cast<std::size_t>(horizon_count) * bitmap_words, 0u);

  result.node_count = brain.node_count;

  // Census the tissue first. An ablation of nothing looks exactly like an
  // ablation with no effect, and only this number tells them apart.
  {
    std::uint32_t* totals = nullptr;
    if (cudaMalloc(&totals, sizeof(std::uint32_t) * 3) == cudaSuccess) {
      cudaMemset(totals, 0, sizeof(std::uint32_t) * 3);
      // The atomicMin's identity, so "no inhibitory tissue" is distinguishable
      // from "the first inhibitory node is node 0".
      cudaMemset(totals + 2, 0xff, sizeof(std::uint32_t));
      census_inhibitory_kernel<<<grid, block>>>(brain.nodes, brain.routes, brain.node_count,
                                                brain.route_capacity, totals, totals + 1,
                                                totals + 2);
      std::uint32_t host_totals[3] = {0u, 0u, 0u};
      if (cudaGetLastError() == cudaSuccess &&
          cudaMemcpy(host_totals, totals, sizeof(host_totals), cudaMemcpyDeviceToHost) ==
              cudaSuccess) {
        result.inhibitory_nodes = host_totals[0];
        result.inhibitory_routes = host_totals[1];
        result.first_inhibitory_node = host_totals[2];
      }
      cudaFree(totals);
    }
  }

  std::uint32_t* device_bitmap = nullptr;
  std::uint32_t* device_count = nullptr;
  unsigned long long* device_digest = nullptr;
  int* device_census = nullptr;
  if (cudaMalloc(&device_bitmap, sizeof(std::uint32_t) * bitmap_words) != cudaSuccess)
    return result;
  if (cudaMalloc(&device_count, sizeof(std::uint32_t)) != cudaSuccess) {
    cudaFree(device_bitmap);
    return result;
  }
  if (cudaMalloc(&device_digest, sizeof(unsigned long long)) != cudaSuccess) {
    cudaFree(device_bitmap);
    cudaFree(device_count);
    return result;
  }
  if (cudaMalloc(&device_census, sizeof(int) * 4) != cudaSuccess) {
    cudaFree(device_bitmap);
    cudaFree(device_count);
    cudaFree(device_digest);
    return result;
  }

  // `seed` == kNoSeed is the BASELINE pass: an identical clone stepped through
  // identical horizons with nothing injected. Everything reported below is the
  // difference between an injected run and this one, so a population number
  // means drive attributable to the seed rather than whatever the organism was
  // already doing.
  constexpr std::uint32_t kNoSeed = 0xffffffffu;
  for (std::uint32_t pass = 0u; pass <= seed_count && ok; ++pass) {
    const std::uint32_t seed = pass == 0u ? kNoSeed : pass - 1u;

    DirectBrain clone{};
    if (!clone_brain_device(brain, &clone, &result.clone_pointer_repairs,
                            &result.clone_pointers_outside_arena,
                            &result.clone_pools_deep_copied)) {
      ok = false;
      break;
    }
    // The lesion is applied to EVERY clone including the baseline pass, so the
    // unstimulated control is lesioned identically and the difference between
    // them stays attributable to the contact rather than to the lesion.
    if (config.ablate_inhibitory_sign || config.ablate_inhibition_threshold) {
      std::uint32_t* touched = nullptr;
      if (cudaMalloc(&touched, sizeof(std::uint32_t) * 3) == cudaSuccess) {
        cudaMemset(touched, 0, sizeof(std::uint32_t) * 3);
        cudaMemset(touched + 2, 0xff, sizeof(std::uint32_t));
        ablate_inhibition_kernel<<<grid, block>>>(
            clone.nodes, clone.routes, clone.node_count, clone.route_capacity,
            result.inhibitory_nodes,
            result.first_inhibitory_node == kNoInhibitoryTissue ? 0u : result.first_inhibitory_node,
            config.ablate_inhibitory_sign, config.ablate_inhibition_threshold,
            config.sham_ablation, touched, touched + 1, touched + 2);
        std::uint32_t host_touched[3] = {0u, 0u, 0u};
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(host_touched, touched, sizeof(host_touched), cudaMemcpyDeviceToHost) ==
                cudaSuccess) {
          result.lesioned_nodes = host_touched[0];
          result.lesioned_routes = host_touched[1];
          result.lesion_first_node = host_touched[2];
        }
        cudaFree(touched);
      }
    }

    // Seed choice is an observer-only intervention on the isolated clone.
    // A real adult ingress event still requires a sensor port, so install one
    // clone-local measurement port rather than bypassing the membrane check.
    DirectBoundaryPort* probe_port = nullptr;
    if (seed != kNoSeed &&
        !install_probe_sensor_port(&clone, result.seed_node[seed], seed,
                                   &probe_port)) {
      destroy_clone(&clone, brain);
      ok = false;
      break;
    }
    DirectAdultRuntime* runtime = create_direct_adult_runtime(&clone, execution);
    if (runtime == nullptr) {
      release_probe_sensor_port(&clone, probe_port);
      destroy_clone(&clone, brain);
      ok = false;
      break;
    }

    if (seed != kNoSeed) {
      ActivityEvent event{};
      event.ticket_id = 0xB0510000ull + seed;
      event.node = result.seed_node[seed];
      event.channel = seed;
      event.word = 0x9E5Bu + seed;
      event.origin = CausalOrigin::external_contact;
      event.context = 0u;
      bool injected = true;
      for (std::uint32_t c = 0u; c < contacts && injected; ++c) {
        // Distinct ticket and word per contact: repeating one identity would
        // settle the same ticket rather than deliver a new contact.
        event.ticket_id = 0xB0510000ull + static_cast<std::uint64_t>(seed) * 1024ull + c;
        event.word = 0x9E5Bu + seed * 7u + c;
        event.timestamp = 1u + c;
        // The instrument control. If the queue rejected this, everything below
        // would be a measurement of a contact that never happened.
        injected = inject_sensory_event(runtime, event);
        if (injected) ++result.events_injected;
      }
      if (!injected) {
        destroy_direct_adult_runtime(runtime);
        release_probe_sensor_port(&clone, probe_port);
        destroy_clone(&clone, brain);
        ok = false;
        break;
      }
    }

    // Horizon zero, on the baseline pass only: what gestation handed over.
    if (seed == kNoSeed) {
      cudaMemset(device_count, 0, sizeof(std::uint32_t));
      cudaMemset(device_digest, 0, sizeof(unsigned long long));
      const int birth_init[4] = {0x7fffffff, -0x7fffffff - 1, 0x7fffffff, -0x7fffffff - 1};
      cudaMemcpy(device_census, birth_init, sizeof(birth_init), cudaMemcpyHostToDevice);
      cudaMemset(device_bitmap, 0, sizeof(std::uint32_t) * bitmap_words);
      capture_population_kernel<<<grid, block>>>(clone.nodes, clone.node_count,
                                                 config.activation_threshold_q16,
                                                 device_bitmap, device_count);
      digest_state_kernel<<<grid, block>>>(clone.nodes, clone.node_count, device_digest,
                                           device_census);
      int birth[4] = {0, 0, 0, 0};
      cudaMemcpy(&result.birth_population, device_count, sizeof(std::uint32_t),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(birth, device_census, sizeof(birth), cudaMemcpyDeviceToHost);
      result.birth_activation_min_q16 = birth[0];
      result.birth_activation_max_q16 = birth[1];
      result.birth_attractor_support_min_q16 = birth[2];
      result.birth_attractor_support_max_q16 = birth[3];
    }

    std::uint32_t stepped = 0u;
    for (std::uint32_t h = 0u; h < horizon_count; ++h) {
      const std::uint32_t target = config.horizon_ticks[h];
      if (target > stepped) {
        step_direct_adult_fixed_morphology_epochs(runtime, target - stepped);
        stepped = target;
      }
      cudaMemset(device_bitmap, 0, sizeof(std::uint32_t) * bitmap_words);
      cudaMemset(device_count, 0, sizeof(std::uint32_t));
      cudaMemset(device_digest, 0, sizeof(unsigned long long));
      const int census_init[4] = {0x7fffffff, -0x7fffffff - 1, 0x7fffffff, -0x7fffffff - 1};
      cudaMemcpy(device_census, census_init, sizeof(census_init), cudaMemcpyHostToDevice);
      capture_population_kernel<<<grid, block>>>(clone.nodes, clone.node_count,
                                                 config.activation_threshold_q16,
                                                 device_bitmap, device_count);
      digest_state_kernel<<<grid, block>>>(clone.nodes, clone.node_count, device_digest,
                                           device_census);
      unsigned long long digest = 0ull;
      if (cudaMemcpy(&digest, device_digest, sizeof(digest), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        ok = false;
        break;
      }
      if (seed == kNoSeed) {
        result.baseline_state_digest[h] = digest;
        if (h + 1u == horizon_count) {
          int census[4] = {0, 0, 0, 0};
          cudaMemcpy(census, device_census, sizeof(census), cudaMemcpyDeviceToHost);
          result.baseline_activation_min_q16 = census[0];
          result.baseline_activation_max_q16 = census[1];
          result.baseline_attractor_support_min_q16 = census[2];
          result.baseline_attractor_support_max_q16 = census[3];
        }
      } else {
        result.seed_state_digest[seed][h] = digest;
      }
      std::vector<std::uint32_t> bits(bitmap_words, 0u);
      if (cudaGetLastError() != cudaSuccess ||
          cudaMemcpy(bits.data(), device_bitmap, sizeof(std::uint32_t) * bitmap_words,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        ok = false;
        break;
      }

      if (seed == kNoSeed) {
        std::uint32_t population = 0u;
        for (std::uint32_t w = 0u; w < bitmap_words; ++w) {
          baseline_bits[static_cast<std::size_t>(h) * bitmap_words + w] = bits[w];
          population += popcount32(bits[w]);
        }
        result.baseline_population[h] = population;
        continue;
      }

      std::uint32_t raw = 0u;
      std::uint32_t attributable = 0u;
      for (std::uint32_t w = 0u; w < bitmap_words; ++w) {
        raw += popcount32(bits[w]);
        bits[w] &= ~baseline_bits[static_cast<std::size_t>(h) * bitmap_words + w];
        attributable += popcount32(bits[w]);
      }
      result.raw_population[seed][h] = raw;
      result.population[seed][h] = attributable;

      for (std::uint32_t w = 0u; w < bitmap_words; ++w)
        ever_active_bits[static_cast<std::size_t>(seed) * bitmap_words + w] |= bits[w];

      const std::uint32_t seed_word = result.seed_node[seed] >> 5;
      const std::uint32_t seed_bit = 1u << (result.seed_node[seed] & 31u);
      if ((bits[seed_word] & seed_bit) != 0u) result.seed_active_mask[seed] |= 1u << h;
      if (h + 1u == horizon_count)
        for (std::uint32_t w = 0u; w < bitmap_words; ++w)
          final_population[static_cast<std::size_t>(seed) * bitmap_words + w] = bits[w];
    }

    const substrate::direct_adult_core::AdultCoreMetrics metrics =
        substrate::direct_adult_core::get_adult_core_metrics(runtime);
    if (seed != kNoSeed)
      result.events_ingested_by_law += metrics.sensory_events_ingested;
    else
      result.baseline_events_ingested_by_law += metrics.sensory_events_ingested;

    {
      int* peak = nullptr;
      if (cudaMalloc(&peak, sizeof(int)) == cudaSuccess) {
        const int seed_value = result.max_slow_context_q16;
        cudaMemcpy(peak, &seed_value, sizeof(int), cudaMemcpyHostToDevice);
        max_slow_context_kernel<<<grid, block>>>(runtime->node_slow_context_q16,
                                                 clone.node_count, peak);
        int host_peak = 0;
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(&host_peak, peak, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess)
          result.max_slow_context_q16 = host_peak;
        cudaFree(peak);
      }
    }

    destroy_direct_adult_runtime(runtime);
    release_probe_sensor_port(&clone, probe_port);
    destroy_clone(&clone, brain);
  }
  cudaFree(device_bitmap);
  cudaFree(device_count);
  cudaFree(device_digest);
  cudaFree(device_census);
  if (!ok) return result;

  const std::uint32_t last = horizon_count - 1u;
  for (std::uint32_t s = 0u; s < seed_count; ++s) {
    if (result.seed_state_digest[s][last] != result.baseline_state_digest[last])
      ++result.seeds_that_changed_state;
    std::uint32_t ever = 0u;
    for (std::uint32_t w = 0u; w < bitmap_words; ++w)
      ever += popcount32(ever_active_bits[static_cast<std::size_t>(s) * bitmap_words + w]);
    result.ever_active[s] = ever;
  }

  // ---- territory sizes, so a reachable count reads as a fraction ---------
  {
    std::uint32_t* sizes = nullptr;
    const std::size_t bytes = sizeof(std::uint32_t) * BasinProbeResult::kMaxCensusTerritories;
    if (cudaMalloc(&sizes, bytes) == cudaSuccess) {
      cudaMemset(sizes, 0, bytes);
      census_territory_sizes_kernel<<<grid, block>>>(brain.nodes, brain.node_count, sizes);
      if (cudaGetLastError() == cudaSuccess)
        cudaMemcpy(result.territory_node_count, sizes, bytes, cudaMemcpyDeviceToHost);
      cudaFree(sizes);
    }
  }

  // ---- tree-wide route baseline (#1293) ---------------------------------
  //
  // Measured, and the header's own numbers recorded beside it. The two are kept
  // separate on purpose: `long_tract_count` in DirectBrain is a field
  // gestation writes, and a field is not evidence that the flag it summarises
  // was ever stamped on a route. Only a count of the tissue says that, and only
  // a comparison says whether the header is telling the truth.
  result.declared_active_route_count = brain.active_route_count;
  result.declared_long_tract_count = brain.long_tract_count;
  {
    std::uint32_t* totals = nullptr;
    const std::size_t bytes = sizeof(std::uint32_t) * 3u;
    if (cudaMalloc(&totals, bytes) == cudaSuccess) {
      cudaMemset(totals, 0, bytes);
      census_route_totals_kernel<<<grid, block>>>(brain.nodes, brain.routes, brain.node_count,
                                                  brain.route_capacity, totals, totals + 1u,
                                                  totals + 2u);
      std::uint32_t host_totals[3] = {};
      if (cudaGetLastError() == cudaSuccess &&
          cudaMemcpy(host_totals, totals, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
        result.measured_active_routes = host_totals[0];
        result.measured_long_tract_routes = host_totals[1];
        result.measured_inhibitory_routes = host_totals[2];
      }
      cudaFree(totals);
    }
  }

  // Long tracts and inhibitory routes leaving each seed node personally,
  // against the active routes leaving the same node. All three come off the
  // same slice in the same pass for the same reason the reachable-set census
  // does.
  for (std::uint32_t s = 0u; s < seed_count; ++s) {
    DirectNode node{};
    if (cudaMemcpy(&node, brain.nodes + result.seed_node[s], sizeof(DirectNode),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      continue;
    const std::uint64_t end = static_cast<std::uint64_t>(node.route_offset) + node.route_capacity;
    if (end > brain.route_capacity) continue;
    std::vector<DirectRoute> slice(node.route_capacity);
    if (node.route_capacity == 0u ||
        cudaMemcpy(slice.data(), brain.routes + node.route_offset,
                   sizeof(DirectRoute) * node.route_capacity, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
      continue;
    for (const DirectRoute& route : slice) {
      if ((route.flags & kRouteFlagActive) == 0u) continue;
      ++result.seed_active_routes[s];
      if ((route.flags & kRouteFlagLongTract) != 0u) ++result.seed_long_tracts[s];
      if ((route.flags & kRouteFlagInhibitory) != 0u) ++result.seed_inhibitory_routes[s];
    }
  }

  // ---- reachability census, on the ORIGINAL graph ------------------------
  //
  // The route graph is never modified by the probe, so this runs on `brain`
  // directly rather than on a clone; the observer-only arm still holds because
  // nothing here writes organism memory.
  std::vector<std::uint32_t> reachable_final(
      static_cast<std::size_t>(seed_count) * bitmap_words, 0u);
  {
    std::uint32_t* reached = nullptr;
    std::uint32_t* frontier = nullptr;
    std::uint32_t* next_frontier = nullptr;
    std::uint32_t* added = nullptr;
    const std::size_t map_bytes = sizeof(std::uint32_t) * bitmap_words;
    if (cudaMalloc(&reached, map_bytes) == cudaSuccess &&
        cudaMalloc(&frontier, map_bytes) == cudaSuccess &&
        cudaMalloc(&next_frontier, map_bytes) == cudaSuccess &&
        cudaMalloc(&added, sizeof(std::uint32_t)) == cudaSuccess) {
      for (std::uint32_t s = 0u; s < seed_count; ++s) {
        cudaMemset(reached, 0, map_bytes);
        cudaMemset(frontier, 0, map_bytes);
        seed_bitmap_kernel<<<1, 1>>>(reached, result.seed_node[s]);
        seed_bitmap_kernel<<<1, 1>>>(frontier, result.seed_node[s]);
        std::uint32_t total = 1u;  // the seed itself
        std::uint32_t hop = 0u;
        for (std::uint32_t h = 0u; h < horizon_count; ++h) {
          while (hop < config.horizon_ticks[h]) {
            cudaMemset(next_frontier, 0, map_bytes);
            cudaMemset(added, 0, sizeof(std::uint32_t));
            expand_reachable_kernel<<<grid, block>>>(brain.nodes, brain.routes, brain.node_count,
                                                     brain.route_capacity, frontier, reached,
                                                     next_frontier, added);
            std::uint32_t newly = 0u;
            if (cudaGetLastError() != cudaSuccess ||
                cudaMemcpy(&newly, added, sizeof(newly), cudaMemcpyDeviceToHost) != cudaSuccess)
              break;
            total += newly;
            // gh #1347: next_frontier already holds this hop's result; swap the
            // pointers instead of copying map_bytes device-to-device. Safe
            // because both buffers are reused ping-pong style across every hop
            // and only freed (once each, regardless of which variable holds
            // which allocation) after the whole seed loop ends.
            std::uint32_t* frontier_swap = frontier;
            frontier = next_frontier;
            next_frontier = frontier_swap;
            ++hop;
            if (newly == 0u) {
              hop = config.horizon_ticks[horizon_count - 1u];  // closed, no more growth
              break;
            }
          }
          result.reachable[s][h] = total;
        }
        cudaMemcpy(reachable_final.data() + static_cast<std::size_t>(s) * bitmap_words, reached,
                   map_bytes, cudaMemcpyDeviceToHost);

        std::uint32_t* census = nullptr;
        // [0..K) per-territory, [K] long tracts, [K+1] active routes (the
        // matched denominator), [K+2] inhibitory routes (#1300).
        const std::size_t census_bytes =
            sizeof(std::uint32_t) * (BasinProbeResult::kMaxCensusTerritories + 3u);
        if (cudaMalloc(&census, census_bytes) == cudaSuccess) {
          cudaMemset(census, 0, census_bytes);
          census_reachable_kernel<<<grid, block>>>(
              brain.nodes, brain.routes, brain.node_count, brain.route_capacity, reached, census,
              census + BasinProbeResult::kMaxCensusTerritories,
              census + BasinProbeResult::kMaxCensusTerritories + 1u,
              census + BasinProbeResult::kMaxCensusTerritories + 2u);
          std::uint32_t host_census[BasinProbeResult::kMaxCensusTerritories + 3u] = {};
          if (cudaGetLastError() == cudaSuccess &&
              cudaMemcpy(host_census, census, census_bytes, cudaMemcpyDeviceToHost) ==
                  cudaSuccess) {
            for (std::uint32_t k = 0u; k < BasinProbeResult::kMaxCensusTerritories; ++k) {
              result.reachable_per_territory[s][k] = host_census[k];
              if (host_census[k] != 0u) ++result.reachable_territory_count[s];
            }
            result.reachable_long_tracts[s] =
                host_census[BasinProbeResult::kMaxCensusTerritories];
            result.reachable_active_routes[s] =
                host_census[BasinProbeResult::kMaxCensusTerritories + 1u];
            result.reachable_inhibitory_routes[s] =
                host_census[BasinProbeResult::kMaxCensusTerritories + 2u];
          }
          cudaFree(census);
        }
      }
    }
    cudaFree(reached);
    cudaFree(frontier);
    cudaFree(next_frontier);
    cudaFree(added);
  }

  for (std::uint32_t a = 0u; a < seed_count; ++a) {
    for (std::uint32_t b = a + 1u; b < seed_count; ++b) {
      std::uint32_t shared = 0u;
      std::uint32_t union_size = 0u;
      for (std::uint32_t w = 0u; w < bitmap_words; ++w) {
        const std::uint32_t left =
            reachable_final[static_cast<std::size_t>(a) * bitmap_words + w];
        const std::uint32_t right =
            reachable_final[static_cast<std::size_t>(b) * bitmap_words + w];
        shared += popcount32(left & right);
        union_size += popcount32(left | right);
      }
      if (union_size == 0u) continue;
      ++result.reachable_compared_pairs;
      if (shared == 0u)
        ++result.reachable_disjoint_pairs;
      else
        ++result.reachable_overlapping_pairs;
      if (result.reachable_compared_pairs == 1u || shared < result.min_reachable_pair_shared)
        result.min_reachable_pair_shared = shared;
    }
  }

  for (std::uint32_t s = 0u; s < seed_count; ++s) {
    result.recurrent_return[s] =
        mask_shows_return(result.seed_active_mask[s], horizon_count) ? 1u : 0u;
    result.recurrent_basin_count += result.recurrent_return[s];
  }

  for (std::uint32_t a = 0u; a < seed_count; ++a) {
    for (std::uint32_t b = a + 1u; b < seed_count; ++b) {
      std::uint32_t shared = 0u;
      std::uint32_t union_size = 0u;
      for (std::uint32_t w = 0u; w < bitmap_words; ++w) {
        const std::uint32_t left = final_population[static_cast<std::size_t>(a) * bitmap_words + w];
        const std::uint32_t right = final_population[static_cast<std::size_t>(b) * bitmap_words + w];
        shared += popcount32(left & right);
        union_size += popcount32(left | right);
      }
      // A pair of two empty populations is not a disjoint pair; it is two
      // measurements that found nothing, and counting it would let a silent
      // organism satisfy the disjointness requirement.
      if (union_size == 0u) continue;
      ++result.compared_pairs;
      result.pair_shared_matrix[a][b] = shared;
      result.pair_shared_matrix[b][a] = shared;
      if (shared == 0u)
        ++result.disjoint_pairs;
      else
        ++result.overlapping_pairs;
      const std::uint32_t jaccard = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(shared) * 65536ull) / union_size);

      // THE SPLIT (#1280 A). The verdict below is unchanged and still reads the
      // totals; this only records which question each pair was asking, so a
      // zero can never again be read against a denominator that includes pairs
      // incapable of answering it.
      const bool same_territory = result.seed_territory[a] == result.seed_territory[b];
      if (same_territory) {
        ++result.same_territory_pairs;
        if (shared == 0u)
          ++result.same_territory_disjoint;
        else
          ++result.same_territory_overlapping;
        if (result.same_territory_pairs == 1u) {
          result.same_min_pair_shared = shared;
          result.same_max_pair_shared = shared;
        } else {
          if (shared < result.same_min_pair_shared) result.same_min_pair_shared = shared;
          if (shared > result.same_max_pair_shared) result.same_max_pair_shared = shared;
        }
      } else {
        ++result.cross_territory_pairs;
        if (shared == 0u)
          ++result.cross_territory_disjoint;
        else
          ++result.cross_territory_overlapping;
        if (result.cross_territory_pairs == 1u) {
          result.cross_min_pair_shared = shared;
          result.cross_max_pair_shared = shared;
          result.cross_min_pair_jaccard_q16 = jaccard;
          result.cross_max_pair_jaccard_q16 = jaccard;
        } else {
          if (shared < result.cross_min_pair_shared) result.cross_min_pair_shared = shared;
          if (shared > result.cross_max_pair_shared) result.cross_max_pair_shared = shared;
          if (jaccard < result.cross_min_pair_jaccard_q16)
            result.cross_min_pair_jaccard_q16 = jaccard;
          if (jaccard > result.cross_max_pair_jaccard_q16)
            result.cross_max_pair_jaccard_q16 = jaccard;
        }
      }

      if (result.compared_pairs == 1u) {
        result.min_pair_shared = shared;
        result.max_pair_shared = shared;
        result.min_pair_jaccard_q16 = jaccard;
        result.max_pair_jaccard_q16 = jaccard;
      } else {
        if (shared < result.min_pair_shared) result.min_pair_shared = shared;
        if (shared > result.max_pair_shared) result.max_pair_shared = shared;
        if (jaccard < result.min_pair_jaccard_q16) result.min_pair_jaccard_q16 = jaccard;
        if (jaccard > result.max_pair_jaccard_q16) result.max_pair_jaccard_q16 = jaccard;
      }
    }
  }
  for (std::uint32_t s = 0u; s < seed_count; ++s) {
    const std::uint32_t population = result.population[s][last];
    if (s == 0u || population < result.min_final_population)
      result.min_final_population = population;
    if (population > result.max_final_population) result.max_final_population = population;
  }
  result.probe_ran = 1u;
  return result;
}

NetworkFoundationReceipt certify_direct_juvenile_with_basins(
    const certification::JuvenileReplica* replicas, std::uint32_t replica_count,
    const BasinProbeConfig& config, const AdultExecutionConfig& execution,
    BasinProbeResult* out_probe) {
  NetworkFoundationReceipt receipt =
      certification::certify_direct_juvenile(replicas, replica_count, config.block_size);
  if (replicas == nullptr || replica_count == 0u || replicas[0].brain == nullptr) return receipt;

  const BasinProbeResult probe = probe_basins(*replicas[0].brain, config, execution);
  if (out_probe != nullptr) *out_probe = probe;

  // A probe that could not run leaves t2 exactly as certify_direct_juvenile
  // left it: unassayed. Scoring a failed instrument as a failed organism is the
  // same error as scoring an unasked question as a passed one.
  if (probe.probe_ran == 0u) return receipt;

  std::uint32_t assayed = certification::kT2ProbeSeedsPostHocFromMatureTissue |
                          certification::kT2MultipleHorizonsThroughLocalLaw |
                          certification::kT2SeveralRecurrentReturnBasins |
                          certification::kT2DisjointAndOverlappingPopulations;
  std::uint32_t unmet = 0u;

  // Seeds are post hoc only if they were actually derived from grown state and
  // the clone they were probed on was faithful.
  if (config.authored_index_seeds || probe.clone_pointers_outside_arena != 0u)
    unmet |= certification::kT2ProbeSeedsPostHocFromMatureTissue;
  if (probe.horizon_count < 3u) unmet |= certification::kT2MultipleHorizonsThroughLocalLaw;
  if (probe.recurrent_basin_count < kSeveralBasins)
    unmet |= certification::kT2SeveralRecurrentReturnBasins;
  if (probe.disjoint_pairs == 0u || probe.overlapping_pairs == 0u)
    unmet |= certification::kT2DisjointAndOverlappingPopulations;

  certification::score_stage(receipt.stage[2], 2u, certification::kT2RequirementCount, assayed,
                             unmet);
  receipt.stage[2].measured[0] = probe.recurrent_basin_count;
  receipt.stage[2].measured[1] = probe.disjoint_pairs;
  receipt.stage[2].measured[2] = probe.overlapping_pairs;
  receipt.stage[2].measured[3] = probe.compared_pairs;
  receipt.stage[2].measured[4] = probe.seed_count;
  receipt.stage[2].measured[5] = probe.horizon_count;
  receipt.stage[2].measured[6] = probe.population[0][probe.horizon_count - 1u];
  receipt.stage[2].measured[7] = probe.clone_pointers_outside_arena;
  // #1280 A: the pair split travels in the RECEIPT, not only in this contract's
  // stdout. `measured[3]` (compared_pairs) was the denominator that made
  // `disjoint_pairs=0` misreadable; a reader of the receipt alone can now see
  // how many of those pairs spanned two territories and how many of THOSE were
  // disjoint, without re-running anything.
  receipt.stage[2].measured[8] = probe.cross_territory_pairs;
  receipt.stage[2].measured[9] = probe.cross_territory_disjoint;
  certification::recompute_certification(receipt);
  return receipt;
}

}  // namespace substrate::direct_network::basin_probe
