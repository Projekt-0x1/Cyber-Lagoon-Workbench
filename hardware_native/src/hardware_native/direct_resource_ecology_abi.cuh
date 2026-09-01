#ifndef HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_ABI_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace substrate::direct_adult {

#if defined(__CUDACC__)
#define DIRECT_ECOLOGY_HD __host__ __device__
#else
#define DIRECT_ECOLOGY_HD
#endif

enum class DirectResourcePoolKind : std::uint16_t {
  node_state = 0,
  explicit_interaction = 1,
  implicit_exception = 2,
  eligibility_record = 3,
  context_record = 4,
  topology_proposal = 5,
  derivation_record = 6,
  derivation_parent_edge = 7,
  representation_source_state = 8,
  packed_panel = 9,
  dense_tile = 10,
  tract_packet = 11,
  delayed_packet = 12,
  checkpoint_future_state = 13,
  // The grown brain's boundary ports: one per body binding that actually
  // attaches. Bindings can fail to attach (unknown seed, inactive territory,
  // out-of-range local node), so this pool's reserved remainder is exactly the
  // count of bindings the body asked for and the organism could not honour.
  boundary_port = 14,
  // A motor act the organism has emitted and whose consequence has not come back
  // yet. The adult runtime holds a fixed table of these, and until this row
  // existed that table was charged as anonymous bytes because no row named it --
  // which is the same defect as a row that names nothing, seen from the other
  // side. `node_state` looked like its home and is not: gestation already charges
  // that pool at capacity == live == node_count for the grown DirectNode records.
  pending_consequence_ticket = 15,
  // #1179 P0.1: the Level B state-owner table (DirectRepresentationStateOwner
  // [cap]) is a distinct unit from `representation_source_state` above -- one
  // entry per live-or-free logical-interaction owner slot, not one per node --
  // so it gets its own pool rather than being folded into
  // representation_source_state and silently meaning two different things.
  // Both are real cudaMalloc silicon created once at runtime creation, never
  // grown incrementally; a table bounded by cudaMalloc but invisible to
  // global_capacity_bytes is bounded in isolation, not bounded with the rest
  // of the organism -- the same defect `pending_consequence_ticket` above
  // closed for the action-return table, unfixed here until now because
  // nothing charged `representation_source_state` either despite it already
  // existing in this enum.
  representation_state_owner = 16,
  count = 17
};

enum DirectResourcePressureFlags : std::uint32_t {
  kDirectPressureNone = 0u,
  kDirectPressureCapacity = 1u << 0,
  kDirectPressureWork = 1u << 1,
  kDirectPressureChurn = 1u << 2,
  kDirectPressureRepair = 1u << 3,
};

enum class DirectRetentionPolicyKind : std::uint32_t {
  rich = 0u,
  minimal = 1u,
};

enum DirectRetentionFlags : std::uint32_t {
  kDirectRetentionNone = 0u,
  kDirectRetentionHardPinned = 1u << 0,
  kDirectRetentionDamaged = 1u << 1,
  kDirectRetentionQuietProtected = 1u << 2,
  kDirectRetentionInTransition = 1u << 3,
  kDirectRetentionRepaired = 1u << 4,
  kDirectRetentionObsolete = 1u << 5,
};

enum DirectHardPinReason : std::uint32_t {
  kPinNone = 0u,
  kPinUnresolvedEligibility = 1u << 0,
  kPinRepresentationMigration = 1u << 1,
  kPinTopologyTransaction = 1u << 2,
  kPinInFlightPacket = 1u << 3,
  kPinRepairPending = 1u << 4,
};

// One pool of physically scarce matter.
//
// `charged_units` is the single admission gate: it is the only word consulted
// when deciding whether matter may be created, and it is incremented before any
// physical object exists. `live_units` and `reserved_units` are the *breakdown*
// of that charge; they exist for reporting and reconciliation, never for
// admission. Splitting the gate across two words is what made the earlier
// protocol racy: a committer that decremented `reserved` before incrementing
// `live` left a window in which a unit was counted in neither, and a concurrent
// reserver reading `live + reserved` looked straight through it and over-admitted.
//
// Invariants (asserted by cuda_direct_resource_physical_authority_contract):
//   charged_units <= capacity_units                             ALWAYS
//   live_units + reserved_units == charged_units                at quiescence
//   live_units == independently counted physical live objects   at quiescence
//
// Turnover moves a unit between `live` and `reserved` (commit/uncommit) and
// leaves `charged_units` alone; only reclaim (release/cancel) decharges. So a
// pool inside one arena allocation can FALL in `live` while `charged` stays
// pinned at what the allocation actually costs -- which is the truth, because
// retracting a route frees no silicon.
struct DirectResourcePoolState {
  std::uint64_t capacity_units;
  std::uint64_t charged_units;
  std::uint64_t live_units;
  std::uint64_t reserved_units;
  std::uint64_t high_water_units;  // atomic max over charged_units, not live_units
  std::uint64_t rejected_units;
  std::uint64_t reclaimed_units;
  std::uint64_t deferred_units;
  std::uint64_t compaction_units;
  std::uint64_t bytes_per_unit;  // physical bytes one unit of this pool occupies
};
static_assert(std::is_trivial_v<DirectResourcePoolState> &&
              std::is_standard_layout_v<DirectResourcePoolState>);

struct DirectWorkBudgetState {
  std::uint64_t logical_epoch;
  std::uint64_t frontier_events;
  std::uint64_t interaction_evaluations;
  std::uint64_t eligibility_lookups;
  std::uint64_t topology_proposals;
  std::uint64_t topology_commits;
  std::uint64_t tract_packets;
  std::uint64_t dense_tile_ops;
  std::uint64_t maintenance_records_visited;
  std::uint64_t maintenance_mutations;
  std::uint64_t deferred_work;
};
static_assert(std::is_trivial_v<DirectWorkBudgetState> &&
              std::is_standard_layout_v<DirectWorkBudgetState>);

struct DirectStructuralChurnState {
  std::uint64_t epoch_window_begin;
  std::uint64_t proposals;
  std::uint64_t admitted;
  std::uint64_t rejected_capacity;
  std::uint64_t rejected_policy;
  std::uint64_t deferred;
  std::uint64_t committed;
  std::uint64_t reclaimed;
  std::uint64_t repaired;
  std::uint64_t representation_moves;
  std::uint64_t mutation_budget;
};
static_assert(std::is_trivial_v<DirectStructuralChurnState> &&
              std::is_standard_layout_v<DirectStructuralChurnState>);

struct DirectRepresentationCostObservation {
  std::uint64_t persistent_bytes;
  std::uint64_t resident_state_bytes;
  std::uint64_t evaluations;
  std::uint64_t dram_bytes_estimate;
  std::uint64_t mutation_count;
  std::uint64_t exceptions;
  std::uint64_t last_window_tick;
};
static_assert(std::is_trivial_v<DirectRepresentationCostObservation> &&
              std::is_standard_layout_v<DirectRepresentationCostObservation>);

// Structure for rich retention policy
struct alignas(16) DirectRetentionState {
  std::uint32_t logical_source;
  std::uint32_t logical_slot;
  std::uint64_t logical_generation;
  std::int32_t support_ema_q16;
  std::uint32_t usage_ema_q16;
  std::uint32_t contradiction_ema_q16;
  std::uint32_t repair_evidence_q16;
  std::int32_t last_confirmed_conductance_q16;
  std::uint64_t last_use_tick;
  std::uint64_t quiet_protect_until;
  std::uint64_t last_support_tick;
  std::uint64_t source_revision;
  std::uint32_t flags;
  std::uint32_t pin_reasons;
};
static_assert(std::is_trivial_v<DirectRetentionState> &&
              std::is_standard_layout_v<DirectRetentionState>);

// Structure for minimal retention policy (#1200)
struct alignas(8) DirectMinimalRetentionState {
  std::uint32_t activity_ema_q16;
  std::uint32_t mismatch_count;
  std::uint32_t last_use_tick;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectMinimalRetentionState> &&
              std::is_standard_layout_v<DirectMinimalRetentionState>);

struct DirectResourceEcologyState {
  DirectResourcePoolState pools[static_cast<std::uint32_t>(DirectResourcePoolKind::count)];
  DirectWorkBudgetState work;
  DirectStructuralChurnState churn;
  // Per-pool capacity buys starvation/fairness isolation, but fourteen
  // independently bounded pools whose capacities sum past the real arena are
  // fourteen virtual budgets, not silicon accounting. The global byte budget is
  // the authoritative physical bound; every reservation is charged against it in
  // the same failure-atomic step as its pool charge.
  std::uint64_t global_capacity_bytes;
  std::uint64_t global_charged_bytes;
  std::uint64_t global_high_water_bytes;
  std::uint64_t global_rejected_bytes;
  std::uint64_t maintenance_cursor;
  // The context index gets its own rolling cursor. It is a separate arena with a
  // separate capacity, so sharing the route cursor would tie the revisit period of
  // one to the size of the other.
  std::uint64_t context_maintenance_cursor;
  std::uint64_t context_reclaimed_bindings;
  std::uint64_t maintenance_epoch;
  std::uint64_t maintenance_scan_budget;
  // Bounds the records inspected per candidate, so a bounded candidate count is
  // also a bounded cost. Without it, "scan 256 routes" can mean 256 x
  // eligibility_count record reads.
  std::uint64_t eligibility_scan_bound;
  // Structural cap lives in churn.mutation_budget.
  std::uint64_t policy_revision;
  DirectRetentionPolicyKind retention_policy;
  std::uint32_t pressure_flags;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectResourceEcologyState> &&
              std::is_standard_layout_v<DirectResourceEcologyState>);

struct DirectRepresentationTransitionProposal {
  std::uint32_t source;
  std::uint32_t route_slot;
  std::uint64_t generation;
  std::uint32_t from_representation;
  std::uint32_t preferred_to_representation;
  DirectRepresentationCostObservation measured;
  std::uint64_t policy_revision;
};
static_assert(std::is_trivial_v<DirectRepresentationTransitionProposal> &&
              std::is_standard_layout_v<DirectRepresentationTransitionProposal>);


// ---------------------------------------------------------------------------
// Allocation authority primitives.
//
// These live in the ABI header, not in direct_resource_ecology.cuh, precisely so
// that the *real* physical allocators (direct_dynamic_topology_arena.cu and
// friends) can charge the ledger without including the adult brain header. A
// ledger the allocators cannot reach is telemetry; only a ledger every allocator
// must pass through is a law.
//
// Transactional shape, mandatory for every physical allocation:
//
//   if (!device_reserve_pool_units(eco, pool, n)) return fail_closed;
//   handle = allocate_physical_slot(...);
//   if (!handle.valid) { device_cancel_pool_reservation(eco, pool, n); return fail_closed; }
//   device_commit_pool_units(eco, pool, n);
//
// and for every reclaim, exactly once per physically retired object:
//
//   if (retire_physical_object_exactly_once(handle)) device_release_pool_units(eco, pool, n);
#if defined(__CUDACC__)

__device__ inline unsigned long long* direct_ecology_u64(std::uint64_t* p) {
  return reinterpret_cast<unsigned long long*>(p);
}

__device__ inline unsigned long long direct_ecology_atomic_add_u64(std::uint64_t* p,
                                                                   std::uint64_t v) {
  return atomicAdd(direct_ecology_u64(p), static_cast<unsigned long long>(v));
}

// CUDA has no 64-bit atomicSub; subtracting is adding the two's-complement.
__device__ inline unsigned long long direct_ecology_atomic_sub_u64(std::uint64_t* p,
                                                                   std::uint64_t v) {
  return atomicAdd(direct_ecology_u64(p),
                   static_cast<unsigned long long>(-static_cast<long long>(v)));
}

__device__ inline DirectResourcePoolState* direct_ecology_pool(
    DirectResourceEcologyState* ecology, DirectResourcePoolKind kind) {
  if (ecology == nullptr) return nullptr;
  const std::uint32_t idx = static_cast<std::uint32_t>(kind);
  if (idx >= static_cast<std::uint32_t>(DirectResourcePoolKind::count)) return nullptr;
  return &ecology->pools[idx];
}

// Reserve capacity for matter that does not exist yet. Fails closed.
//
// A single atomic on `charged_units` is the whole admission decision, so the
// answer cannot be assembled from two words that another thread is midway
// through updating. The global byte budget is charged in the same failure-atomic
// step: if either bound is exceeded, both charges are rolled back before
// returning, so a rejected reservation leaves no phantom exhaustion behind.
__device__ inline bool device_reserve_pool_units(DirectResourceEcologyState* ecology,
                                                 DirectResourcePoolKind kind,
                                                 std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return false;

  const unsigned long long prev = direct_ecology_atomic_add_u64(&pool->charged_units, units);
  if (prev + units > pool->capacity_units) {
    direct_ecology_atomic_sub_u64(&pool->charged_units, units);
    direct_ecology_atomic_add_u64(&pool->rejected_units, units);
    atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(kDirectPressureCapacity));
    return false;
  }

  const std::uint64_t bytes = units * pool->bytes_per_unit;
  if (bytes != 0u && ecology->global_capacity_bytes != 0u) {
    const unsigned long long gprev =
        direct_ecology_atomic_add_u64(&ecology->global_charged_bytes, bytes);
    if (gprev + bytes > ecology->global_capacity_bytes) {
      direct_ecology_atomic_sub_u64(&ecology->global_charged_bytes, bytes);
      direct_ecology_atomic_sub_u64(&pool->charged_units, units);
      direct_ecology_atomic_add_u64(&pool->rejected_units, units);
      direct_ecology_atomic_add_u64(&ecology->global_rejected_bytes, bytes);
      atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(kDirectPressureCapacity));
      return false;
    }
    atomicMax(direct_ecology_u64(&ecology->global_high_water_bytes),
              static_cast<unsigned long long>(gprev + bytes));
  }

  direct_ecology_atomic_add_u64(&pool->reserved_units, units);
  atomicMax(direct_ecology_u64(&pool->high_water_units),
            static_cast<unsigned long long>(prev + units));
  return true;
}

// Publish a reservation: the physical object now exists.
//
// Commit CAN REFUSE, and that is not a formality. A void commit that blindly
// subtracts is an unsigned underflow waiting to happen: commit one unit more
// than was ever reserved and `reserved_units` wraps to ~1.8e19, at which point
// the pool reports astronomically more free reserve than the arena physically
// holds and every later admission decision is nonsense. So commit gets a single
// admission word of its own, exactly as reserve() gates on `charged_units`.
//
// That word is `live_units`, and it is gated UPWARD: `live` may never exceed
// `charged`. Gating upward rather than draining `reserved` downward is not a
// stylistic choice, it is the whole cost of the primitive.
//
//   - A downward gate has to be a CAS retry loop, because "subtract only if the
//     result stays non-negative" cannot be expressed as one unsigned add: the
//     obvious add-then-check-then-roll-back races, since a thread that observes
//     an already-wrapped 1.8e19 sees a value that passes the check.
//   - An upward gate CAN be one add, because addition on these magnitudes cannot
//     wrap: take the unit, look at what the atomic returned, put it back if the
//     total went past `charged`. That is the identical shape reserve() already
//     uses against `capacity_units`.
//
// Measured, not assumed: the CAS version was written first and cost 7x. Gestation
// commits one unit per materialized route, so 524,544 threads hit one address;
// same-address atomicAdd is aggregated in hardware, while every CAS retry is a
// full round trip. 65k-node gestation went 100ms -> 708ms and
// cuda_direct_network_life_function_full_contract went RED on its own timing
// budget. The gate below is two adds in the success path -- the same count the
// original unchecked commit had -- and gestation returned to ~100ms.
//
// `charged_units` is untouched — commit moves matter between columns, it does
// not create it. The transient state between the two words is an over-count (the
// unit is briefly counted in both), never an under-count, so no concurrent
// reader can see a unit vanish. Only the post-synchronize state is contracted;
// no reconciliation may read these words from inside a running kernel.
__device__ inline bool device_commit_pool_units(DirectResourceEcologyState* ecology,
                                                DirectResourcePoolKind kind,
                                                std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return false;
  const unsigned long long previous = direct_ecology_atomic_add_u64(&pool->live_units, units);
  if (previous + units > pool->charged_units) {
    direct_ecology_atomic_sub_u64(&pool->live_units, units);
    atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(kDirectPressureCapacity));
    return false;
  }
  direct_ecology_atomic_sub_u64(&pool->reserved_units, units);
  return true;
}

// Return committed matter to the reserve WITHOUT decharging it.
//
// This is the turnover primitive for matter that lives inside one arena
// allocation. When the grown brain retracts a route it does not hand silicon
// back to the driver: the slot is still in the arena, still paid for, still
// exactly as expensive. What changed is that it stopped being live. Calling
// release() here would be a lie about a single cudaMalloc -- it would decrement
// `charged_units` for memory nothing freed, and the pool would slowly report
// capacity the organism does not have.
//
// So uncommit is the exact inverse of commit, and it is gated the same way for
// the same cost reason: `reserved_units` may never exceed `charged_units`, which
// is precisely the statement that `live_units` may never go negative. One add,
// one comparison, one rollback on the failure path -- no CAS loop. The grown
// brain retracts hundreds of thousands of routes in a single run, so this is a
// hot path too, not an exception handler.
__device__ inline bool device_uncommit_pool_units(DirectResourceEcologyState* ecology,
                                                  DirectResourcePoolKind kind,
                                                  std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return false;
  const unsigned long long previous = direct_ecology_atomic_add_u64(&pool->reserved_units, units);
  if (previous + units > pool->charged_units) {
    direct_ecology_atomic_sub_u64(&pool->reserved_units, units);
    return false;
  }
  direct_ecology_atomic_sub_u64(&pool->live_units, units);
  // Deliberately NOT `reclaimed_units`. Reclaim means the silicon went back to
  // the driver and `charged_units` fell with it; conflating the two would let a
  // brain that only ever recycles in place look like it was returning memory.
  // `compaction_units` was a column nothing wrote until now, which is what this
  // event always was: matter recycled inside an allocation that never moved.
  direct_ecology_atomic_add_u64(&pool->compaction_units, units);
  return true;
}

// Roll back a reservation whose physical allocation failed or was abandoned.
// Without this the reserve path leaks capacity and the pool starves forever.
__device__ inline void device_cancel_pool_reservation(DirectResourceEcologyState* ecology,
                                                      DirectResourcePoolKind kind,
                                                      std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return;
  direct_ecology_atomic_sub_u64(&pool->reserved_units, units);
  direct_ecology_atomic_sub_u64(&pool->charged_units, units);
  const std::uint64_t bytes = units * pool->bytes_per_unit;
  if (bytes != 0u && ecology->global_capacity_bytes != 0u)
    direct_ecology_atomic_sub_u64(&ecology->global_charged_bytes, bytes);
}

// Release committed matter. Must be called exactly once per physically retired
// object, after that object is unreachable.
__device__ inline void device_release_pool_units(DirectResourceEcologyState* ecology,
                                                 DirectResourcePoolKind kind,
                                                 std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return;
  direct_ecology_atomic_sub_u64(&pool->live_units, units);
  direct_ecology_atomic_sub_u64(&pool->charged_units, units);
  direct_ecology_atomic_add_u64(&pool->reclaimed_units, units);
  const std::uint64_t bytes = units * pool->bytes_per_unit;
  if (bytes != 0u && ecology->global_capacity_bytes != 0u)
    direct_ecology_atomic_sub_u64(&ecology->global_charged_bytes, bytes);
}

// Record a request refused on policy grounds (no capacity was ever charged).
__device__ inline void device_reject_pool_units(DirectResourceEcologyState* ecology,
                                                DirectResourcePoolKind kind,
                                                std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return;
  direct_ecology_atomic_add_u64(&pool->rejected_units, units);
  atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(kDirectPressureCapacity));
}

// Record work postponed to a later tick rather than performed or refused.
__device__ inline void device_defer_pool_units(DirectResourceEcologyState* ecology,
                                               DirectResourcePoolKind kind,
                                               std::uint64_t units) {
  DirectResourcePoolState* pool = direct_ecology_pool(ecology, kind);
  if (pool == nullptr || units == 0u) return;
  direct_ecology_atomic_add_u64(&pool->deferred_units, units);
}

#endif  // __CUDACC__

#undef DIRECT_ECOLOGY_HD

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_ABI_CUH
