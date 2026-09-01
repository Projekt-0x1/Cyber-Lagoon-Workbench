#include "hardware_native/direct_resource_ecology_legacy.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace substrate::direct_adult {
namespace {

void check_res_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// Claim one maintenance window.
//
// The window start MUST be produced before the scanning kernel launches. The
// earlier design had every scanning thread read `maintenance_cursor` while
// thread (0,0) of the same kernel advanced it, with no ordering between them:
// threads in later blocks could read the already-advanced cursor and scan the
// *next* window, leaving part of the current one permanently unvisited. That is
// not a slow scan, it is a scan that silently never maintains some of the brain.
//
// Claiming with a single atomicAdd and passing the result down as a kernel
// argument makes coverage exact: stride is 1 and windows are contiguous, so every
// slot is visited exactly once per ceil(capacity / budget) steps regardless of
// whether the capacity is prime, a power of two, or an odd composite.
__global__ void claim_maintenance_window_kernel(DirectResourceEcologyState* ecology,
                                                std::uint32_t route_capacity,
                                                std::uint32_t context_capacity,
                                                std::uint32_t actual_scan,
                                                std::uint64_t* window_start_out) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || ecology == nullptr) return;
  const unsigned long long claimed =
      atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->maintenance_cursor),
                static_cast<unsigned long long>(actual_scan));
  window_start_out[0] = route_capacity == 0u ? 0ull : (claimed % route_capacity);
  const unsigned long long context_claimed =
      atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->context_maintenance_cursor),
                static_cast<unsigned long long>(actual_scan));
  window_start_out[1] = context_capacity == 0u ? 0ull : (context_claimed % context_capacity);
  // pressure_flags accumulates via atomicOr from every allocator on the device.
  // Clearing it here, single-threaded and before the scan, is the one point at
  // which a plain store is correct; the old code did a plain store *after*
  // allocators had already OR-ed into it, silently discarding their reports.
  ecology->pressure_flags = kDirectPressureNone;
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->maintenance_epoch), 1ull);
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.maintenance_records_visited),
            static_cast<unsigned long long>(actual_scan));
}

__global__ void unified_rolling_maintenance_kernel(
    DirectBrainV01 brain,
    DirectRetentionState* retention_bank,
    DirectMinimalRetentionState* minimal_bank,
    const DirectEligibilityRecord* eligibility_bank,
    std::uint32_t eligibility_count,
    DirectResourceMaintenanceRuntime maintenance_rt,
    std::uint64_t current_tick,
    std::uint32_t actual_scan,
    const std::uint64_t* window_start_ptr,
    const std::uint32_t* eligibility_count_device) {
  DirectResourceEcologyState* ecology = brain.resource_ecology;
  if (ecology == nullptr || brain.route_capacity == 0u) return;

  // 0X1-1175: clamp the host-supplied record count to the device-resident live
  // count. This is the same shape the topology epoch uses for its live bound --
  // the device reads the count, the host never does -- so it costs no readback.
  // It matters because `eligibility_count` decides the *limit* of the pin scan
  // below AND the `limit < eligibility_count` test that turns a truncated scan
  // into `DirectPinLookup::unknown`. `unknown` folds to "pinned" on purpose,
  // because deferring a reclaim is the safe direction; but that conservatism is
  // only safe when the count is true. Fed a launch bound over an empty bank, it
  // reports EVERY live route hard-pinned, every tick, forever.
  std::uint32_t live_eligibility = eligibility_count;
  if (eligibility_count_device != nullptr) {
    const std::uint32_t live = *eligibility_count_device;
    live_eligibility = live < eligibility_count ? live : eligibility_count;
  }

  // 1. Thread 0 of block 0 reports non-capacity pressure.
  //
  // NOTE what is deliberately absent here: the old code recomputed
  //   route_pool.live_units = route_capacity - *free_count
  // on every maintenance step. That single line is what made the whole ledger a
  // shadow. It derived the ledger from the allocator instead of the allocator
  // passing through the ledger, so it clobbered every real reserve/commit charge
  // and made a capacity rejection impossible on the live path. `live_units` is
  // now written ONLY by commit/release in direct_resource_ecology_abi.cuh, from
  // the physical allocators themselves, and is reconciled against an independent
  // count of live slot_meta entries by
  // cuda_direct_resource_physical_authority_contract.
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    if (brain.topology.free_count != nullptr) {
      const std::uint32_t free_c = *brain.topology.free_count;
      if (free_c < (brain.route_capacity / 10u + 1u))
        atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(kDirectPressureCapacity));
    }
  }

  // 2a. Bounded rolling reclaim of the context index.
  //
  // A context-index entry binds a (source, signature) key to a physical route.
  // When that route is retracted the binding becomes garbage: lookup already
  // refuses to return it, but nothing ever freed it, so the pool could only fill
  // and the organism would eventually lose the ability to index any new context.
  //
  // The binding, not the table slot, is the unit the ledger charges, so returning
  // `route` to kInvalidIndex is exactly the release that matches the charge. The
  // slot keeps its source/signature, which preserves the open-addressed probe
  // chain -- clearing those would strand every key that probed past this slot --
  // and the entry can be re-bound later by the same key.
  const std::uint32_t ctx_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ctx_idx < actual_scan && brain.context_index != nullptr &&
      brain.context_index_capacity != 0u) {
    const std::uint32_t ctx_slot = static_cast<std::uint32_t>(
        (window_start_ptr[1] + ctx_idx) % brain.context_index_capacity);
    ContextRouteIndexEntry& entry = brain.context_index[ctx_slot];
    const std::uint32_t bound = entry.route;
    if (bound != kInvalidIndex) {
      bool dead = bound >= brain.route_capacity;
      if (!dead) {
        const DirectRouteSlotMeta bound_meta = brain.topology.slot_meta[bound];
        dead = (bound_meta.live == 0u) || (bound_meta.generation != entry.route_generation);
      }
      // The CAS is what makes the release exactly-once: only the thread that
      // actually unbinds the entry books the release.
      if (dead && atomicCAS(&entry.route, bound, kInvalidIndex) == bound) {
        __threadfence();
        entry.route_generation = 0u;
        device_release_pool_units(ecology, DirectResourcePoolKind::context_record, 1u);
        atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->context_reclaimed_bindings),
                  1ull);
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &ecology->pools[static_cast<std::uint32_t>(
                                          DirectResourcePoolKind::context_record)]
                           .compaction_units),
                  1ull);
      }
    }
  }

  // 2b. Parallel bounded rolling scan over the claimed route window.
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < actual_scan) {
    const std::uint64_t start_cursor = window_start_ptr[0];
    const std::uint32_t slot = static_cast<std::uint32_t>((start_cursor + idx) % brain.route_capacity);
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[slot];

    if (maintenance_rt.visit_counts != nullptr)
      atomicAdd(&maintenance_rt.visit_counts[slot], 1u);

    if (meta.live != 0u) {
      if (maintenance_rt.candidate_slots != nullptr && idx < maintenance_rt.scan_capacity)
        maintenance_rt.candidate_slots[idx] = slot;

      std::uint32_t lookups = 0u;
      const std::uint32_t scan_bound =
          static_cast<std::uint32_t>(ecology->eligibility_scan_bound);
      const DirectPinLookup pin = is_route_hard_pinned_by_eligibility_exact(
          eligibility_bank, live_eligibility, slot, meta.generation, scan_bound, &lookups);
      const bool pinned = pin != DirectPinLookup::not_pinned;
      if (lookups > 0u)
        atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.eligibility_lookups),
                  static_cast<unsigned long long>(lookups));
      if (pin == DirectPinLookup::unknown)
        atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.deferred_work), 1ull);

      if (ecology->retention_policy == DirectRetentionPolicyKind::rich && retention_bank != nullptr) {
        DirectRetentionState& ret = retention_bank[slot];

        // Retention state is stored per *slot*, but it describes a route, and a
        // route is a (slot, generation) pair. If the stored generation is not the
        // one currently living here, the record belongs to a route that no longer
        // exists.
        //
        // Skipping such a record is not enough -- that was the first version of
        // this gate and the authority contract caught it. Skipping while still
        // stamping the current generation onto the record LAUNDERS it: the stale
        // flags survive, the identity is refreshed, and the very next sweep sees a
        // "current" damaged record and resurrects a dead generation's conductance.
        // A record for a route that no longer exists carries no information about
        // the one that does, so it is discarded outright.
        const bool generation_current = (ret.logical_generation == meta.generation);
        if (!generation_current) {
          DirectRetentionState fresh{};
          fresh.logical_source = brain.routes[slot].source;
          fresh.logical_slot = slot;
          fresh.logical_generation = meta.generation;
          fresh.last_confirmed_conductance_q16 = brain.routes[slot].conductance_q16;
          if (pinned) {
            fresh.flags |= kDirectRetentionHardPinned;
            fresh.pin_reasons |= kPinUnresolvedEligibility;
          }
          ret = fresh;
        } else {
          if (pinned) {
            ret.flags |= kDirectRetentionHardPinned;
            ret.pin_reasons |= kPinUnresolvedEligibility;
          } else {
            ret.flags &= ~kDirectRetentionHardPinned;
            ret.pin_reasons &= ~kPinUnresolvedEligibility;
          }

          if ((ret.flags & kDirectRetentionDamaged) != 0u && ret.support_ema_q16 > 0) {
            // Case A repair: degraded but still physically owned. The slot never
            // left `live`, so no matter is being created and no pool charge is
            // due -- repair here restores content, not existence. A repair that
            // followed an actual physical release would have to reserve afresh
            // like any other allocation; the authority contract asserts
            // live_units is unchanged across a repair-only step so that this
            // distinction cannot quietly rot into free resurrection.
            ret.flags |= kDirectRetentionRepaired;
            ret.flags &= ~kDirectRetentionDamaged;
            brain.routes[slot].conductance_q16 = ret.last_confirmed_conductance_q16 > 0
                                                    ? ret.last_confirmed_conductance_q16
                                                    : (1 << 16);
            atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.repaired), 1ull);
            atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.maintenance_mutations),
                      1ull);
          }

          ret.logical_source = brain.routes[slot].source;
          ret.logical_slot = slot;
        }
      } else if (minimal_bank != nullptr) {
        DirectMinimalRetentionState& min_ret = minimal_bank[slot];
        if (pinned) {
          min_ret.flags |= kDirectRetentionHardPinned;
        } else {
          min_ret.flags &= ~kDirectRetentionHardPinned;
        }
      }
    }
  }
}


// Independent physical census: counts live slot_meta entries with no reference to
// any ledger counter. Deriving the check from the thing being checked would make
// the reconciliation vacuous.
__global__ void count_physical_live_routes_kernel(const DirectRouteSlotMeta* slot_meta,
                                                  std::uint32_t route_capacity,
                                                  std::uint32_t* out_count) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= route_capacity) return;
  if (slot_meta[idx].live != 0u) atomicAdd(out_count, 1u);
}

// One-shot seeding at birth/restore: books a charge for matter that already
// physically exists. Uses the same charged/live columns the allocators use, so
// the ledger starts life already reconciled.
__global__ void seed_route_pool_from_matter_kernel(DirectBrainV01 brain) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  DirectResourceEcologyState* ecology = brain.resource_ecology;
  if (ecology == nullptr || brain.topology.slot_meta == nullptr) return;
  std::uint64_t live = 0u;
  for (std::uint32_t i = 0; i < brain.route_capacity; ++i)
    if (brain.topology.slot_meta[i].live != 0u) ++live;
  DirectResourcePoolState& pool =
      ecology->pools[static_cast<std::uint32_t>(DirectResourcePoolKind::explicit_interaction)];
  pool.live_units = live;
  pool.reserved_units = 0u;
  pool.charged_units = live;
  if (live > pool.high_water_units) pool.high_water_units = live;
  if (ecology->global_capacity_bytes != 0u)
    ecology->global_charged_bytes = live * pool.bytes_per_unit;
}

}  // namespace

void initialize_direct_resource_ecology_state(
    DirectResourceEcologyState* state, DirectRetentionPolicyKind policy) {
  if (state == nullptr) return;
  *state = DirectResourceEcologyState{};
  state->retention_policy = policy;
  state->maintenance_scan_budget = 64u;
  state->eligibility_scan_bound = 256u;
  state->churn.mutation_budget = 512u;
  state->policy_revision = 1u;
}

DirectResourceMaintenanceRuntime* create_resource_maintenance_runtime(std::uint32_t scan_capacity) {
  if (scan_capacity == 0u) scan_capacity = 256u;
  DirectResourceMaintenanceRuntime* rt = new DirectResourceMaintenanceRuntime{};
  rt->scan_capacity = scan_capacity;
  check_res_cuda(cudaMalloc(&rt->candidate_slots, sizeof(std::uint32_t) * scan_capacity),
                 "allocate maintenance candidate slots");
  check_res_cuda(cudaMalloc(&rt->candidate_actions, sizeof(std::uint32_t) * scan_capacity),
                 "allocate maintenance candidate actions");
  check_res_cuda(cudaMalloc(&rt->hard_pin_reasons, sizeof(std::uint32_t) * scan_capacity),
                 "allocate maintenance hard pin reasons");
  check_res_cuda(cudaMalloc(&rt->winner_count, sizeof(std::uint32_t)),
                 "allocate maintenance winner count");
  check_res_cuda(cudaMalloc(&rt->window_start, sizeof(std::uint64_t) * 2u),
                 "allocate maintenance window start");
  check_res_cuda(cudaMemset(rt->window_start, 0, sizeof(std::uint64_t) * 2u),
                 "clear maintenance window start");
  return rt;
}

void destroy_resource_maintenance_runtime(DirectResourceMaintenanceRuntime* runtime) {
  if (runtime == nullptr) return;
  if (runtime->window_start != nullptr) cudaFree(runtime->window_start);
  if (runtime->winner_count != nullptr) cudaFree(runtime->winner_count);
  if (runtime->hard_pin_reasons != nullptr) cudaFree(runtime->hard_pin_reasons);
  if (runtime->candidate_actions != nullptr) cudaFree(runtime->candidate_actions);
  if (runtime->candidate_slots != nullptr) cudaFree(runtime->candidate_slots);
  delete runtime;
}

void launch_direct_resource_maintenance_step(
    DirectBrainV01* brain,
    DirectRetentionState* retention_bank,
    DirectMinimalRetentionState* minimal_bank,
    const DirectEligibilityRecord* eligibility_bank,
    std::uint32_t eligibility_count,
    DirectResourceMaintenanceRuntime* maintenance_rt,
    std::uint64_t current_tick,
    std::uint64_t scan_budget,
    cudaStream_t stream,
    std::uint32_t block_size,
    const std::uint32_t* eligibility_count_device) {
  if (brain == nullptr || brain->resource_ecology == nullptr || scan_budget == 0u || block_size == 0u)
    return;

  const std::uint32_t actual_scan = std::min(static_cast<std::uint32_t>(scan_budget), brain->route_capacity);
  const std::uint32_t grid = (actual_scan + block_size - 1u) / block_size;
  DirectResourceMaintenanceRuntime local_rt =
      maintenance_rt != nullptr ? *maintenance_rt : DirectResourceMaintenanceRuntime{};

  // The window must be claimed by a completed kernel before any scanning thread
  // reads it. Two launches on the same stream give that ordering for free.
  std::uint64_t* window_start = local_rt.window_start;
  if (window_start == nullptr) {
    static std::uint64_t* fallback_window = nullptr;
    if (fallback_window == nullptr) {
      check_res_cuda(cudaMalloc(&fallback_window, sizeof(std::uint64_t) * 2u),
                     "allocate fallback maintenance window start");
      check_res_cuda(cudaMemset(fallback_window, 0, sizeof(std::uint64_t) * 2u),
                     "clear fallback maintenance window start");
    }
    window_start = fallback_window;
  }

  claim_maintenance_window_kernel<<<1, 32, 0, stream>>>(
      brain->resource_ecology, brain->route_capacity, brain->context_index_capacity, actual_scan,
      window_start);

  unified_rolling_maintenance_kernel<<<std::max(1u, grid), block_size, 0, stream>>>(
      *brain, retention_bank, minimal_bank, eligibility_bank, eligibility_count,
      local_rt, current_tick, actual_scan, window_start, eligibility_count_device);
}

void seed_resource_pool_ledgers_from_matter(DirectBrainV01* brain, cudaStream_t stream) {
  if (brain == nullptr || brain->resource_ecology == nullptr) return;
  seed_route_pool_from_matter_kernel<<<1, 32, 0, stream>>>(*brain);
  check_res_cuda(cudaGetLastError(), "seed resource pool ledgers from matter");
}

void count_physical_live_routes(const DirectBrainV01* brain, std::uint32_t* device_out,
                                cudaStream_t stream) {
  if (brain == nullptr || device_out == nullptr) return;
  check_res_cuda(cudaMemsetAsync(device_out, 0, sizeof(std::uint32_t), stream),
                 "clear physical live route count");
  if (brain->route_capacity == 0u || brain->topology.slot_meta == nullptr) return;
  const std::uint32_t block = 256u;
  const std::uint32_t grid = (brain->route_capacity + block - 1u) / block;
  count_physical_live_routes_kernel<<<grid, block, 0, stream>>>(
      brain->topology.slot_meta, brain->route_capacity, device_out);
  check_res_cuda(cudaGetLastError(), "count physical live routes");
}

}  // namespace substrate::direct_adult
