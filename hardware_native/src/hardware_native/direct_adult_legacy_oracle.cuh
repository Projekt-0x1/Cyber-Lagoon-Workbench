// Direct mathematical-network Brain v0.1.
//
// Gamma is compiled once by a massively parallel Life Function into this
// execution-oriented state.  The born state contains no Genome pointer and
// no developmental interpreter: recipe authority ends at birth.  Lifetime
// activity changes route-local conductance/eligibility directly.

#ifndef HARDWARE_NATIVE_DIRECT_MATHEMATICAL_ADULT_CUH
#define HARDWARE_NATIVE_DIRECT_MATHEMATICAL_ADULT_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_recipe.hpp"
#include "hardware_native/direct_resource_ecology_abi.cuh"
#include "hardware_native/direct_representation_compiler_abi.cuh"

#if defined(__CUDACC__)
#define DIRECT_ADULT_HD __host__ __device__
#else
#define DIRECT_ADULT_HD
#endif

// github #1208 item 1: DirectAdultRuntime carries the CUDA stream its step is
// issued on.  `cudaStream_t` is an alias of `CUstream_st*`, so forward-declaring
// the driver's opaque handle at GLOBAL scope (it must not be redeclared inside
// substrate::direct_adult, which would make a distinct, unrelated type) keeps
// this header includable from translation units that never pull in
// <cuda_runtime.h>.
struct CUstream_st;

namespace substrate::direct_adult {

using Gamma = substrate::direct_network::recipe::Genome;
using Root256 = substrate::direct_network::recipe::Root256;
using Word = std::uint32_t;

inline constexpr std::uint32_t kInvalidIndex = 0xffffffffu;
// Enough for independent raw sensor and motor apertures for every byte value.
// The mapping is a body convention; no byte meaning enters the network law.
inline constexpr std::uint32_t kMaxBoundaryBindings = 512u;
inline constexpr std::uint32_t kMaximumDirectFanout = 8u;
inline constexpr std::uint32_t kMaximumResidentFanout =
    0xffffffffu;  // Matter-bounded; no authored per-node cognition cap.
inline constexpr std::uint32_t kEligibilityLaunchExpansion =
    16u;  // Scheduling factor only; not semantic fanout.
inline constexpr std::int32_t kConductanceOneQ16 = 1 << 16;
inline constexpr std::int32_t kEligibilityOneQ16 = 1 << 16;
// #1235: shared with the #1179 packed-cache winner resolver
// (direct_exact_eligibility_device.cuh's resolve_packed_cache_winner), which
// must exclude the same weak routes propagate_sparse_frontier_kernel's own
// canonical scan excludes -- moved out of direct_adult_legacy_oracle.cu's
// anonymous namespace so a second translation unit can share the exact value.
inline constexpr std::int32_t kMinimumConductanceQ16 = 1 << 10;
inline constexpr std::uint32_t kNodeFlagSensor = 1u << 0;
inline constexpr std::uint32_t kNodeFlagMotor = 1u << 1;
inline constexpr std::uint32_t kNodeFlagWorldConsequence = 1u << 2;
// gh #1243: substrate::direct_network::DirectRoute (direct_network_brain.cuh)
// is a separate, non-interchangeable struct that happens to share this name,
// most field names, and this whole flag vocabulary's names -- but NOT the
// bit positions. direct_network carries an extra kRouteFlagActive at bit 0
// that direct_adult has no equivalent of, so every flag below it is offset
// by one bit: direct_network's kRouteFlagLongTract sits at 1u << 1, one bit
// above this one. Reinterpreting a raw .flags value across the two types
// would misread direct_network's unconditionally-set kRouteFlagActive as
// this file's kRouteFlagLongTract, and direct_network's kRouteFlagLongTract
// as this file's kRouteFlagLearnedOutput -- not merely a mislabeled tract.
// No translation unit currently includes both headers (enforced by
// tools/check_direct_route_namespace_isolation.py) and no verified
// conversion path copies .flags between them. Do not assume the bit
// position or the raw flags value carries across a future growth-to-adult
// lowering; translate the semantic flag, not the integer.
inline constexpr std::uint32_t kRouteFlagLongTract = 1u << 0;
inline constexpr std::uint32_t kRouteFlagLearnedOutput = 1u << 1;

enum class BoundaryRole : std::uint32_t {
  sensor = 1u << 0,
  motor = 1u << 1,
  world_consequence = 1u << 2,
};

struct BoundaryBinding {
  std::uint32_t seed_index;
  std::uint32_t local_node;
  std::uint32_t channel;
  Word word;
  std::uint32_t role_mask;
};

struct BodyManifestV0 {
  std::uint32_t abi_version;
  std::uint32_t binding_count;
  BoundaryBinding bindings[kMaxBoundaryBindings];
};
static_assert(std::is_trivial_v<BodyManifestV0> && std::is_standard_layout_v<BodyManifestV0>);

enum class CausalOrigin : std::uint32_t {
  external_contact = 0,
  world_return = 1,
  endogenous_prediction = 2,
  motor_reafference = 3,
};

// Authentication belongs to the particular frontier entry that crossed the
// action-return bridge, never to public ActivityEvent transport data.  The
// bridge is the sole producer of bridge_authenticated_return; every public
// ingress and every endogenous successor is ordinary.
enum class DirectIngressAuthority : std::uint32_t {
  ordinary = 0u,
  bridge_authenticated_return = 1u,
};

DIRECT_ADULT_HD inline constexpr bool is_independent_external(CausalOrigin origin) {
  return origin == CausalOrigin::external_contact || origin == CausalOrigin::world_return ||
         origin == CausalOrigin::motor_reafference;
}

// A bridge may claim an exact eligibility episode, but a record never carries
// a standing authorization window.  The one-event credential is the parallel
// DirectIngressAuthority entry evaluated beside the frontier event.
inline constexpr std::uint32_t kBridgeSettlementUnclaimed = 0u;
inline constexpr std::uint32_t kBridgeSettlementArmed = 1u;

// An armed episode can settle only through the exact authenticated bridge
// frontier entry.  ActivityEvent::origin remains descriptive transport data;
// public ingress can assert it but cannot assert this sidecar.
DIRECT_ADULT_HD inline constexpr bool bridge_settlement_refused(std::uint32_t claim,
                                                                CausalOrigin origin,
                                                                DirectIngressAuthority authority) {
  return claim != kBridgeSettlementUnclaimed &&
         (origin != CausalOrigin::motor_reafference ||
          authority != DirectIngressAuthority::bridge_authenticated_return);
}

struct DirectNode {
  std::uint32_t first_route;
  std::uint32_t route_count;
  std::uint32_t flags;
  std::uint32_t lineage;
  std::uint32_t refractory_until;
  std::uint32_t external_contacts;
  std::uint32_t endogenous_visits;
  std::uint32_t maintenance_q16;
  std::uint32_t output_channel;
  Word output_word;
  std::uint32_t implicit_family;
  // #1179: widened to 64-bit so the packed-cache staleness guard cannot wrap
  // around under long-running adults. Kept 8-byte aligned on purpose (an
  // explicit padding field, not compiler-inserted padding) so every
  // DirectNode{} value-initialization deterministically zeroes it -- this
  // field's raw bytes are hashed wholesale by direct_brain_state_root, and
  // an indeterminate compiler-inserted pad would make that hash depend on
  // allocator garbage instead of only ever on assigned field values.
  std::uint32_t source_revision_pad;
  std::uint64_t source_revision;
};
static_assert(std::is_trivial_v<DirectNode> && std::is_standard_layout_v<DirectNode>);

struct DirectRoute {
  std::uint32_t source;
  std::uint32_t target;
  std::uint32_t next_route;
  std::uint32_t implicit_family;
  std::uint32_t implicit_slot;
  Word learned_output_word;
  std::uint64_t context_signature;
  std::uint64_t eligibility_history;
  std::uint64_t eligibility_root;
  std::int32_t conductance_q16;
  std::int32_t eligibility_q16;
  std::int32_t last_credit_q16;
  std::uint32_t eligibility_context;
  std::uint32_t eligibility_expires;
  std::uint32_t predicted_context;
  std::uint32_t delay;
  std::uint32_t flags;
  // Compatibility/inspection summary only. Exact causal eligibility lives in
  // DirectEligibilityRecord entries; these counters must never be used as the
  // authority for selecting a consequence lineage.
  std::uint32_t eligibility_live_count;
};
static_assert(std::is_trivial_v<DirectRoute> && std::is_standard_layout_v<DirectRoute>);

struct DirectRouteSlotMeta {
  std::uint64_t generation;
  std::uint64_t logical_epoch;
  std::uint32_t logical_rank;
  std::uint32_t live;
};
static_assert(std::is_trivial_v<DirectRouteSlotMeta> &&
              std::is_standard_layout_v<DirectRouteSlotMeta>);

enum class DirectTopologyProposalKind : std::uint32_t {
  none = 0u,
  grow = 1u,
  retract = 2u,
  repair = 3u,
  materialize_implicit = 4u,
};

struct DirectTopologyProposal {
  DirectTopologyProposalKind kind;
  std::uint32_t source;
  std::uint32_t target;
  std::uint32_t route_slot;
  std::uint64_t route_generation;
  std::uint64_t context_signature;
  std::uint64_t eligibility_history;
  std::uint64_t eligibility_root;
  Word learned_output_word;
  std::int32_t conductance_q16;
  std::int32_t credit_q16;
  std::int32_t priority_q16;
  std::uint32_t delay;
  std::uint32_t route_flags;
  std::uint32_t ordinal;
  std::uint32_t implicit_family;
  std::uint32_t implicit_slot;
  std::uint32_t eligibility_context;
  std::uint32_t eligibility_expires;
  std::uint32_t predicted_context;
};
static_assert(std::is_trivial_v<DirectTopologyProposal> &&
              std::is_standard_layout_v<DirectTopologyProposal>);

struct DirectTopologyPersistentState {
  DirectRouteSlotMeta* slot_meta;
  std::uint32_t* free_slots;
  std::uint32_t* free_count;
  std::uint32_t* incoming_degree;
  std::uint64_t* epoch;
  std::uint32_t route_capacity;
  std::uint32_t max_resident_fanout;
};
static_assert(std::is_trivial_v<DirectTopologyPersistentState> &&
              std::is_standard_layout_v<DirectTopologyPersistentState>);

struct DirectImplicitFamily {
  std::uint32_t node_begin;
  std::uint32_t node_count;
  std::uint32_t local_degree;
  std::uint32_t chord_stride;
  std::uint32_t lineage;
  std::uint32_t first_virtual_slot;
  std::uint32_t virtual_slot_count;
  std::uint32_t flags;
  std::int32_t base_conductance_q16;
  std::int16_t coeff_q12[4];
};
static_assert(std::is_trivial_v<DirectImplicitFamily> &&
              std::is_standard_layout_v<DirectImplicitFamily>);

struct DirectRouteHandle {
  std::uint32_t slot;
  std::uint64_t generation;
};
static_assert(std::is_trivial_v<DirectRouteHandle> && std::is_standard_layout_v<DirectRouteHandle>);

enum class EligibilityHorizonClass : std::uint32_t {
  immediate = 0u,
  short_dialogue = 1u,
  multi_turn = 2u,
  asynchronous_return = 3u,
  long_omission = 4u,
};

enum class EligibilityRecordState : std::uint32_t {
  free = 0u,
  live = 1u,
  settled = 2u,
  expired = 3u,
};

struct DirectEligibilityRecord {
  DirectRouteHandle route;
  std::uint32_t source;
  std::uint32_t context;
  std::uint64_t history_signature;
  std::uint64_t participation_root;
  std::uint64_t ticket;
  std::int32_t strength_q16;
  std::uint32_t participation_tick;
  std::uint32_t expiry_tick;
  std::uint32_t predicted_context;
  EligibilityHorizonClass horizon_class;
  EligibilityRecordState state;
  std::uint32_t next_in_bucket;
  // #1184: zero for ordinary episodes, kBridgeSettlementArmed only after the
  // bridge binds the exact motor ticket.  It is intentionally not an
  // authorization bit: authenticated settlement lives on the frontier sidecar.
  std::uint32_t require_motor_reafference;
  // #1179: stable causal identity + current physical backing. `route` above
  // stays the cached explicit-route locator every existing #1176 read/write
  // site already uses; these two trailing fields let apply_return_credit_kernel
  // additionally settle a record whose backing has moved into state-owning
  // implicit storage (locator.kind == implicit_virtual) without becoming a
  // second, parallel ledger. logical_id.value == 0 (the default for every
  // record created before #1179 or never involved in a representation
  // migration) means these fields are simply unused and behavior is
  // unchanged from pre-#1179.
  DirectLogicalInteractionId logical_id;
  DirectInteractionLocator locator;
  // #1166: the `history_signature` the next stream contact out of this source
  // will carry, predicted at mint time. Zero fails OPEN (pre-#1166 behaviour).
  std::uint64_t successor_history_signature;
};
static_assert(std::is_trivial_v<DirectEligibilityRecord> &&
              std::is_standard_layout_v<DirectEligibilityRecord>);

// #1184: one (ticket, cue_node, context) triple DirectCausalActionBridge
// wants marked strict immediately after draining the MotorEvent that
// produced it. Host-uploaded to mark_bridge_tickets_strict_kernel; trivial
// layout so a plain array can be memcpy'd device-side.
struct BridgeTicketMark {
  std::uint64_t ticket;
  std::uint32_t cue_node;
  std::uint32_t context;
};
static_assert(std::is_trivial_v<BridgeTicketMark> && std::is_standard_layout_v<BridgeTicketMark>);

struct DirectImplicitException {
  std::uint64_t key;
  std::int32_t conductance_delta_q16;
  std::uint32_t flags;
  std::uint32_t participation_count;
  std::uint32_t first_participation_tick;
  std::uint32_t last_participation_tick;
  std::uint64_t first_external_root;
  std::uint32_t has_distinct_external_root;
};
static_assert(std::is_trivial_v<DirectImplicitException> &&
              std::is_standard_layout_v<DirectImplicitException>);

struct DirectImplicitPersistentState {
  DirectImplicitFamily* families;
  DirectImplicitException* exceptions;
  std::uint32_t* exception_count;
  std::uint32_t family_count;
  std::uint32_t exception_capacity;
  std::uint64_t virtual_interaction_count;
};
static_assert(std::is_trivial_v<DirectImplicitPersistentState> &&
              std::is_standard_layout_v<DirectImplicitPersistentState>);

struct ContextRouteIndexEntry {
  std::uint64_t signature;
  std::uint64_t route_generation;
  std::uint32_t source;
  std::uint32_t route;
};
static_assert(std::is_trivial_v<ContextRouteIndexEntry> &&
              std::is_standard_layout_v<ContextRouteIndexEntry>);

struct ActivityEvent {
  std::uint32_t node;
  Word word;
  CausalOrigin origin;
  std::uint32_t context;
  std::uint32_t cue_node;
  std::uint32_t source_id;
  std::uint64_t external_root;
  std::uint32_t horizon;
  std::uint64_t history_signature;
};
static_assert(std::is_trivial_v<ActivityEvent> && std::is_standard_layout_v<ActivityEvent>);

struct MotorEvent {
  std::uint32_t node;
  std::uint32_t channel;
  Word word;
  std::uint32_t context;
  std::uint64_t causal_root;
  // #1184: the resolved originating (cue) node of this causal episode. An
  // external action-return bridge cannot address a reafferent return without
  // this -- the exact eligibility ledger indexes by (cue_node, context), not
  // by the motor node the trajectory happened to reach. Trailing/additive so
  // existing 5-field aggregate-init call sites still compile unchanged.
  std::uint32_t cue_node;
};
static_assert(std::is_trivial_v<MotorEvent> && std::is_standard_layout_v<MotorEvent>);

struct ResidentDevelopmentState {
  std::uint32_t developmental_tick;
  std::uint32_t maturation_tick;
  std::uint32_t plasticity_q16;
  std::uint32_t mature_plasticity_floor_q16;
  std::uint64_t constructor_reserve;
  std::uint32_t free_slot_target;
  std::uint64_t reclaimed_resource;
};
static_assert(std::is_trivial_v<ResidentDevelopmentState> &&
              std::is_standard_layout_v<ResidentDevelopmentState>);

struct DirectBrainV01 {
  DirectNode* nodes;
  DirectRoute* routes;
  ResidentDevelopmentState* development;
  std::uint32_t* live_route_count;
  ContextRouteIndexEntry* context_index;
  DirectTopologyPersistentState topology;
  DirectImplicitPersistentState implicit;
  DirectResourceEcologyState* resource_ecology;
  DirectRetentionState* retention_bank;
  DirectMinimalRetentionState* minimal_retention_bank;
  std::uint32_t node_count;
  std::uint32_t route_count;  // physically materialized routes at birth
  std::uint32_t route_capacity;
  std::uint32_t context_index_capacity;
  std::uint32_t territory_count;
  std::uint32_t recurrent_route_count;  // logical recurrent interaction count
  std::uint32_t long_tract_count;
  std::uint64_t logical_route_count;
  std::uint64_t virtual_route_count;
  Root256 genome_root;
  Root256 body_root;
  Root256 birth_root;
};
static_assert(std::is_trivial_v<DirectBrainV01> && std::is_standard_layout_v<DirectBrainV01>);

struct BirthReceiptV0 {
  Root256 genome_root;
  Root256 body_root;
  Root256 birth_root;
  std::uint32_t node_count;
  std::uint32_t route_count;  // logical route/interactions for compatibility
  std::uint32_t territory_count;
  std::uint32_t recurrent_route_count;
  std::uint32_t long_tract_count;
  std::uint64_t device_bytes;
  float gestation_ms;
  bool compact_recipe;
  bool final_connectome_loaded;
  bool life_function_detached;
  std::uint32_t explicit_route_count;
  std::uint32_t implicit_family_count;
  std::uint64_t virtual_route_count;
};

struct AdultCounters {
  std::uint32_t propagated;
  std::uint32_t motor_events;
  std::uint32_t matches;
  std::uint32_t mismatches;
  std::uint32_t omissions;
  std::uint32_t credit_commits;
  std::uint32_t ignored_returns;
  std::uint32_t structural_growth;
  std::uint32_t growth_exhausted;
  std::uint32_t context_index_hits;
  std::uint32_t context_index_inserts;
  std::uint32_t structural_retract;
  std::uint32_t structural_repairs;
  std::uint32_t structural_proposals;
  std::uint32_t structural_duplicate_reject;
  std::uint32_t structural_fanout_reject;
  std::uint32_t structural_target_reject;
  std::uint32_t structural_stale_reject;
  std::uint32_t topology_epochs;
  std::uint32_t implicit_wins;
  std::uint32_t implicit_materializations;
  std::uint32_t eligibility_stale_generation_reject;
  std::uint32_t virtual_participations;
  std::uint32_t packed_hits;
  std::uint32_t dense_activations;
  std::uint32_t tract_admissions;
  std::uint32_t tract_deliveries;
  std::uint32_t guard_fallbacks;
  std::uint32_t eligibility_capacity_rejects;
  std::uint32_t eligibility_exact_settlements;
  std::uint32_t eligibility_ticket_rejects;
  std::uint32_t eligibility_ambiguous_context_rejects;
  std::uint32_t eligibility_duplicate_settlement_rejects;
  std::uint32_t eligibility_index_collisions;
  std::uint32_t eligibility_index_probes;
  std::uint32_t eligibility_expired_records;
  std::uint32_t eligibility_merged_duplicates;
  // #1184: a live episode whose bound route targets a motor node (i.e. an
  // outstanding action ticket a bridge adapter has not yet answered) may
  // only be closed by a CausalOrigin::motor_reafference event. An ordinary
  // CausalOrigin::external_contact credit event that would otherwise have
  // matched such an episode (exact ticket coincidence or the coarse
  // "unique live episode" fallback) is counted here instead of being
  // allowed to settle it -- the authority-boundary rejection is itself a
  // resident-observable transport fact, not silent. Trailing/additive;
  // AdultCounters is memset-zeroed each step so existing call sites are
  // unaffected.
  std::uint32_t eligibility_motor_action_spoof_rejects;
  // #1178: explicit silicon resource ecology counters
  std::uint32_t resource_reclaims;
  std::uint32_t resource_repairs;
  std::uint32_t resource_capacity_rejects;
  std::uint32_t resource_crossovers;
  // #1255: set (after the per-step AdultCounters{} zero-init, inside the
  // same kernel launch) only by clear_adult_step_idle_state_kernel, so a
  // step-loop caller can distinguish "this tick took the topology_work == 0
  // fast path" from a busy tick without inferring it from launch bounds.
  // Trailing/additive; AdultCounters is memset-zeroed each step so existing
  // call sites are unaffected.
  std::uint32_t idle_step_taken;
  // 0X1-1176: successors this tick's propagation ATTEMPTED to append beyond the
  // frontier allocation, i.e. `attempted - capacity`, captured before the count
  // is clamped to residency. The adult attempts about three times its own
  // frontier allocation every tick and the surplus is discarded by `atomicAdd`
  // race order, which is both its saturation and its nondeterminism source.
  // Trailing/additive; AdultCounters is memset-zeroed each step so existing
  // call sites are unaffected.
  std::uint32_t frontier_successors_discarded;
};

struct ResidentContextState {
  std::uint32_t last_external_node;
  Word last_external_word;
  std::uint32_t previous_external_node;
  Word previous_external_word;
  std::uint32_t rolling_history;
  std::uint32_t sequence_length;
  std::uint64_t actual_contact_count;
  std::uint64_t boundary_count;
};
static_assert(std::is_trivial_v<ResidentContextState> &&
              std::is_standard_layout_v<ResidentContextState>);

struct AdultStepReceipt {
  AdultCounters counters;
  std::uint32_t next_frontier_size;
  std::uint32_t motor_count;
  std::uint32_t eligibility_count;
  std::uint32_t pending_tract_packets;
};
static_assert(std::is_trivial_v<AdultStepReceipt> && std::is_standard_layout_v<AdultStepReceipt>);
static_assert(sizeof(AdultStepReceipt) == (sizeof(AdultCounters) + 4u * sizeof(std::uint32_t)));

struct DirectTopologyRuntime;
struct DirectExecutionFabricRuntime;
struct DirectResourceMaintenanceRuntime;
struct DirectRepresentationRuntime;

struct DirectAdultRuntime {
  DirectBrainV01 brain;
  ActivityEvent* frontier;
  ActivityEvent* next_frontier;
  DirectIngressAuthority* frontier_authority;
  DirectIngressAuthority* next_frontier_authority;
  std::uint32_t* frontier_count;
  std::uint32_t* next_frontier_count;
  std::uint32_t frontier_capacity;
  // Persistent host->device staging for batched ingress. Sized to
  // frontier_capacity because a batch larger than the frontier can hold is
  // chunked. Scratch only -- never captured, restored, or rooted.
  ActivityEvent* ingress_staging;
  mutable std::uint32_t frontier_launch_bound;
  // Launch geometry only; it never selects cognitive work.  Tests exercise
  // schedule invariance at 64/128/256 threads while production defaults to 128.
  std::uint32_t return_credit_launch_block;
  MotorEvent* motor_events;
  std::uint32_t* motor_count;
  std::uint32_t motor_capacity;
  // Persistent upload staging for bridge ticket marks.  Marks are always a
  // subset of one step's own drained motor events, so `motor_capacity` bounds
  // them; holding the buffer keeps a cudaMalloc/cudaFree pair off every
  // action-bearing step.  Scratch only -- never captured, restored, or rooted.
  BridgeTicketMark* bridge_mark_staging;
  DirectEligibilityRecord* eligibility_bank;
  DirectEligibilityRecord* next_eligibility_bank;
  std::uint32_t* eligibility_count;
  std::uint32_t* next_eligibility_count;
  std::uint32_t eligibility_capacity;
  std::uint32_t* eligibility_bucket_heads;
  std::uint32_t* next_eligibility_bucket_heads;
  std::uint32_t eligibility_bucket_count;
  std::uint32_t* eligibility_batch_admit;
  mutable std::uint32_t eligibility_launch_bound;
  AdultCounters* counters;
  ResidentContextState* context_state;
  AdultStepReceipt* step_receipt;
  DirectTopologyRuntime* topology_runtime;
  DirectExecutionFabricRuntime* fabric;
  DirectResourceMaintenanceRuntime* resource_maintenance;
  DirectRepresentationRuntime* representation;
  std::uint32_t tick;
  // github #1208 item 1: the CUDA stream every operation this file issues for
  // the step is placed on.  Value-initialised to nullptr, which IS the legacy
  // default stream, so every existing caller keeps byte-identical behaviour and
  // no call site has to be touched.  A caller that wants the step to be
  // graph-capturable assigns its own stream here before capture;
  // operations still issued to the legacy stream by files this runtime calls
  // into (topology arena, fabric, tract, packed, checkpoint) remain the reason
  // capture cannot yet succeed.
  //
  // It MUST be a *blocking* stream (plain cudaStreamCreate).  Ordering between
  // this file's dispatches and the still-legacy dispatches of the files the step
  // calls into -- and the blocking cudaMemcpy in observe_adult_step -- rests
  // entirely on legacy-default-stream implicit synchronisation, which
  // cudaStreamNonBlocking switches off.  A non-blocking stream here would be a
  // silent correctness change, not just a scheduling one.
  //
  // Spelled as `CUstream_st*` (the type `cudaStream_t` is an alias of) rather
  // than `cudaStream_t` so this header stays includable from translation units
  // that never pull in <cuda_runtime.h>; assignment from a cudaStream_t is
  // exact, not a conversion.
  CUstream_st* stream;
};

DIRECT_ADULT_HD inline std::uint32_t context_index_slot(std::uint32_t source,
                                                        std::uint64_t signature,
                                                        std::uint32_t capacity) {
  std::uint64_t value = signature ^ (static_cast<std::uint64_t>(source) * 0x9e3779b97f4a7c15ull);
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return static_cast<std::uint32_t>(value) & (capacity - 1u);
}

DIRECT_ADULT_HD inline std::uint32_t eligibility_bucket_slot(std::uint32_t source,
                                                             std::uint32_t context,
                                                             std::uint64_t ticket,
                                                             std::uint32_t capacity) {
  if (capacity == 0u)
    return 0u;
  (void)ticket;  // Ticket/history/root are exact authority inside the bucket.
  std::uint64_t value = (static_cast<std::uint64_t>(source) << 32) | context;
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return static_cast<std::uint32_t>(value) & (capacity - 1u);
}

#if defined(__CUDACC__)
__device__ inline bool install_context_route(DirectBrainV01 brain, std::uint32_t source,
                                             std::uint64_t signature, std::uint32_t route,
                                             AdultCounters* counters) {
  if (signature == 0u || brain.context_index_capacity == 0u || route >= brain.route_capacity)
    return false;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[route];
  if (meta.live == 0u)
    return false;
  std::uint32_t slot = context_index_slot(source, signature, brain.context_index_capacity);
  for (std::uint32_t probe = 0; probe < 32u; ++probe) {
    ContextRouteIndexEntry& entry = brain.context_index[slot];
    std::uint32_t observed_source = atomicAdd(&entry.source, 0u);
    if (observed_source == kInvalidIndex) {
      // Claim the slot through route while source remains unpublished. This
      // prevents another lane from observing source==source before signature
      // and generation are initialized.
      // The ledger decides before the table does. A context-index slot is real
      // device memory, so claiming one is an allocation and must be reservable,
      // refusable and releasable like any other -- otherwise the index is a pool
      // that grows outside the law while the law watches.
      if (!device_reserve_pool_units(brain.resource_ecology,
                                     DirectResourcePoolKind::context_record, 1u))
        return false;
      const std::uint32_t claimed_route = atomicCAS(&entry.route, kInvalidIndex, route);
      if (claimed_route != kInvalidIndex)
        device_cancel_pool_reservation(brain.resource_ecology,
                                       DirectResourcePoolKind::context_record, 1u);
      if (claimed_route == kInvalidIndex) {
        device_commit_pool_units(brain.resource_ecology,
                                 DirectResourcePoolKind::context_record, 1u);
        entry.signature = signature;
        entry.route_generation = meta.generation;
        __threadfence();
        atomicExch(&entry.source, source);
        if (counters != nullptr)
          atomicAdd(&counters->context_index_inserts, 1u);
        return true;
      }
      // A concurrent initializer owns this slot. Warp divergence guarantees its
      // publication can complete before this bounded observation loop resumes.
      for (std::uint32_t spin = 0u; spin < 64u; ++spin) {
        observed_source = atomicAdd(&entry.source, 0u);
        if (observed_source != kInvalidIndex)
          break;
      }
      if (observed_source == kInvalidIndex)
        return false;
    }
    if (observed_source == source && entry.signature == signature) {
      const std::uint32_t indexed_route = atomicAdd(&entry.route, 0u);
      if (indexed_route == kInvalidIndex)
        return true;  // already marked ambiguous; canonical scan owns selection.
      if (indexed_route == route) {
        entry.route_generation = meta.generation;
        __threadfence();
        if (atomicAdd(&entry.route, 0u) != route)
          entry.route_generation = 0u;
        return true;
      }
      // Two different physical routes are valid for the same acceleration key.
      // The index must not let CUDA schedule order choose one. Mark the entry
      // ambiguous so lookup falls back to the canonical route scan.
      atomicExch(&entry.route, kInvalidIndex);
      __threadfence();
      entry.route_generation = 0u;
      return true;
    }
    slot = (slot + 1u) & (brain.context_index_capacity - 1u);
  }
  return false;
}

__device__ inline std::uint32_t lookup_context_route(DirectBrainV01 brain, std::uint32_t source,
                                                     std::uint64_t signature) {
  if (signature == 0u || brain.context_index_capacity == 0u)
    return kInvalidIndex;
  std::uint32_t slot = context_index_slot(source, signature, brain.context_index_capacity);
  for (std::uint32_t probe = 0; probe < 32u; ++probe) {
    const ContextRouteIndexEntry entry = brain.context_index[slot];
    if (entry.source == kInvalidIndex)
      return kInvalidIndex;
    if (entry.source == source && entry.signature == signature) {
      if (entry.route >= brain.route_capacity)
        return kInvalidIndex;  // ambiguous or stale: canonical scan owns selection.
      const DirectRouteSlotMeta meta = brain.topology.slot_meta[entry.route];
      if (meta.live != 0u && meta.generation == entry.route_generation)
        return entry.route;
      return kInvalidIndex;
    }
    slot = (slot + 1u) & (brain.context_index_capacity - 1u);
  }
  return kInvalidIndex;
}
#endif

BirthReceiptV0 compile_direct_brain_v01(const Gamma& gamma, const BodyManifestV0& body,
                                        DirectBrainV01* out_brain);
void destroy_direct_brain(DirectBrainV01* brain);
DirectAdultRuntime create_direct_adult_runtime(DirectBrainV01 brain,
                                               std::uint32_t frontier_capacity = 4096u,
                                               std::uint32_t motor_capacity = 256u,
                                               std::uint32_t eligibility_capacity = 0u);
void destroy_direct_adult_runtime(DirectAdultRuntime* runtime, bool destroy_brain = false);
// Public membrane ingress. It deliberately does NOT police `event.origin`:
// motor reafference from a body/motor loop that is not the action bridge is a
// real, older, legitimate shape here (cuda_direct_reafferent_byte_stream_
// contract's chronological byte-by-byte prediction closure, cuda_direct_
// multicontext_eligibility_contract, cuda_direct_hybrid_implicit_mesh_
// contract). Origin is a description of the event, not a capability. The
// capability that matters -- settling an episode the action bridge has claimed
// -- is held by DirectEligibilityRecord::require_motor_reafference below, which
// this entry point cannot advance.
void inject_raw_event(DirectAdultRuntime* runtime, const ActivityEvent& event);
// #1184: marks the live eligibility record(s) matching each (cue_node,
// context, ticket) triple as bridge-claimed (kBridgeSettlementArmed) -- see
// DirectEligibilityRecord::require_motor_reafference. Callers must invoke this
// only after the step that opened the record has already completed (i.e. after
// observe_adult_step, exactly when
// DirectCausalActionBridge::drain_pending_actions itself runs) and before
// the next launch_direct_adult_step. A no-op for count == 0.
void mark_bridge_tickets_strict(DirectAdultRuntime* runtime, const BridgeTicketMark* marks,
                                std::uint32_t count);

class DirectCausalActionBridge;

// Capability for the sole privileged ingress path.  It remains user-provided
// (rather than defaulted) so aggregate/value initialization cannot bypass C++
// access control; only DirectCausalActionBridge can construct it.
class BridgeReturnInjectionGrant {
 private:
  BridgeReturnInjectionGrant() {}
  friend class DirectCausalActionBridge;
};

// Private bridge ingress changes authority metadata only; it shares the public
// append path's chronology, context history, and capacity behavior.
// Batched ingress. Semantically identical to calling inject_raw_event once per
// element in order -- the append kernel is single-threaded and the batch kernel
// simply carries the same loop onto the device, so chronology, context history
// and capacity behaviour are unchanged. One launch instead of `count`, which
// matters because issuing a kernel costs ~2.2 us against ~1 us to run it.
void inject_raw_events(DirectAdultRuntime* runtime, const ActivityEvent* events,
                       std::uint32_t count);
void inject_bridge_return_event(DirectAdultRuntime* runtime, const ActivityEvent& event,
                                BridgeReturnInjectionGrant);
void inject_membrane_boundary(DirectAdultRuntime* runtime);
void launch_direct_adult_step(DirectAdultRuntime* runtime);
AdultStepReceipt observe_adult_step(const DirectAdultRuntime& runtime);
Root256 direct_brain_state_root(const DirectBrainV01& brain);

}  // namespace substrate::direct_adult

#undef DIRECT_ADULT_HD

#endif  // HARDWARE_NATIVE_DIRECT_MATHEMATICAL_ADULT_CUH
