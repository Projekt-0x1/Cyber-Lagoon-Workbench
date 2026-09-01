#ifndef HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_ABI_CUH

// #1179 resident representation compiler -- ABI types.
//
// Two production authority levels, kept structurally distinct because they
// answer different questions:
//
//   A. execution-cache crystal: the causal backing (a DirectRoute the normal
//      #1176 ledger already settles against) never moves. A stable
//      small-degree source may additionally get a packed, exact evaluation
//      view for speed. DirectSourceRepresentationState/DirectRepresentationRuntime
//      own this lifecycle; it never touches DirectEligibilityRecord.
//
//   B. state-owning representation: the causal backing itself moves from an
//      explicit DirectRoute slot into procedural (#1187 implicit) storage.
//      A DirectLogicalInteractionId + DirectInteractionLocator let live
//      #1176 eligibility records survive that move without caring which
//      physical form currently backs their logical interaction.
//
// Tensor is not the definition of maturity: a context-conditioned WTA/sparse
// source that fails Tensor's additive-contraction shape is rejected from
// Tensor promotion (counted, not silently dropped) and stays eligible for A.

#include <cstdint>
#include <type_traits>

namespace substrate::direct_adult {

// A stable semantic/causal owner for one logical interaction. value == 0
// means "no #1179 logical identity" -- the interaction never participates in
// representation migration and every #1179 code path leaves it untouched.
struct DirectLogicalInteractionId {
  std::uint64_t value;
};
static_assert(std::is_trivial_v<DirectLogicalInteractionId> &&
              std::is_standard_layout_v<DirectLogicalInteractionId>);

enum class DirectLocatorKind : std::uint32_t {
  explicit_route = 0u,
  implicit_virtual = 1u,
};

// Current physical backing for a logical interaction. For explicit_route,
// (slot, generation) is exactly a DirectRouteHandle into brain.routes /
// brain.topology.slot_meta. For implicit_virtual, slot is an index into
// DirectRepresentationRuntime::state_owners and generation is that record's
// own owner_epoch (bumped on every explicit<->implicit transition so a freed
// state-owner table slot cannot be silently reused underneath a stale
// locator either).
struct DirectInteractionLocator {
  DirectLocatorKind kind;
  std::uint32_t slot;
  std::uint64_t generation;
};
static_assert(std::is_trivial_v<DirectInteractionLocator> &&
              std::is_standard_layout_v<DirectInteractionLocator>);

// -- A. execution-cache crystal -------------------------------------------

enum class DirectPackedLifecycle : std::uint32_t {
  plastic = 0u,        // no candidate; ordinary canonical sparse execution.
  shadow = 1u,          // candidate built; compared against canonical, not yet trusted.
  probation = 2u,       // shadow has agreed enough consecutive touches; armed for promotion.
  active = 3u,           // packed cache is the live evaluation path for this source.
  lesioned = 4u,         // focally disabled; canonical sparse fallback forced.
  tensor_rejected = 5u,  // WTA/sparse shape observed; Tensor promotion explicitly declined.
};

constexpr std::uint32_t kRepresentationResidentReserve = 8u;
constexpr std::uint32_t kRepresentationProbationTouches = 4u;

// #1179 A3: a source whose route membership changes faster than probation
// can absorb pays the FULL rebuild cost of every probation->active promotion
// (walking up to kRepresentationResidentReserve routes) while contributing
// consult hits for only the brief window before the next demotion -- a
// repeatedly churning source is an economically losing refresh, and nothing
// before this counted how many times a given source had already lost that
// bet. `churn_strikes` (capped at kRepresentationChurnStrikeCap) makes each
// subsequent promotion attempt require kRepresentationProbationTouches * (1 +
// churn_strikes) consecutive stable touches instead of the base window, so a
// chronic churner keeps costing canonical-scan cycles for longer before it is
// trusted with another expensive rebuild. `active_stable_touches` counts
// consecutive untouched-by-churn ticks while active; sustaining a full
// backed-off window without a fresh demotion earns one churn_strikes credit
// back, so the penalty decays with demonstrated stability rather than being
// permanent.
constexpr std::uint32_t kRepresentationChurnStrikeCap = 4u;

// Per-source bookkeeping for the packed-cache compiler. One entry per
// DirectNode, indexed identically to brain.nodes. `claim_ordinal` is touched-
// work arbitration scratch: reset every tick for sources present in the
// current frontier and never checkpointed (it has no meaning across a
// restart boundary, only within one tick's arbitration race).
struct DirectSourceRepresentationState {
  DirectPackedLifecycle lifecycle;
  std::uint32_t shadow_agree_streak;
  std::uint64_t shadow_source_revision;
  std::uint32_t claim_ordinal;
  std::uint32_t churn_strikes;
  std::uint32_t active_stable_touches;
};
static_assert(std::is_trivial_v<DirectSourceRepresentationState> &&
              std::is_standard_layout_v<DirectSourceRepresentationState>);

// -- B. state-owning representation ---------------------------------------

enum class DirectStateOwnerLifecycle : std::uint32_t {
  free = 0u,
  shadow = 1u,
  probation_pending_retract = 2u,
  owned = 3u,
  lesioned = 4u,
  rematerialize_pending = 5u,
};

constexpr std::int32_t kRepresentationContradictionStrikeLimit = 3;

// One migrated logical interaction's procedural home. `implicit_credit_accumulator_q16`
// is the representation-independent learned residual (#1179's own state, distinct
// from #1187's procedural family/exception substrate state): on demotion back to
// explicit, exact effective conductance = current procedural substrate conductance
// + this residual.
struct DirectRepresentationStateOwner {
  DirectStateOwnerLifecycle lifecycle;
  DirectLogicalInteractionId logical_id;
  std::uint32_t origin_source;
  std::uint32_t origin_route_slot;
  std::uint64_t origin_route_generation;
  std::uint32_t target;
  std::uint32_t implicit_family;
  std::uint32_t implicit_slot;
  std::uint32_t delay;
  std::uint32_t route_flags;
  std::uint32_t learned_output_word;
  std::uint64_t context_signature;
  std::int64_t implicit_credit_accumulator_q16;
  std::int32_t contradiction_strikes;
  std::uint64_t owner_epoch;
  // Consecutive-tick agreement counter while lifecycle == shadow (armed for
  // probation once it reaches kRepresentationProbationTouches); meaningless
  // once lifecycle reaches probation_pending_retract or later.
  std::uint32_t shadow_streak;
};
static_assert(std::is_trivial_v<DirectRepresentationStateOwner> &&
              std::is_standard_layout_v<DirectRepresentationStateOwner>);

struct DirectRepresentationCounters {
  std::uint32_t packed_cache_promotions;
  std::uint32_t packed_cache_lesions;
  // #1235: bumped when resolve_packed_cache_winner declines an `active`
  // source (a stale entry -- generation mismatch against the live topology
  // arena) and the caller falls back to the canonical linked-list scan for
  // that event. Was declared in #1179 but never wired to anything until the
  // packed cache had a real per-tick consult site to fall back from.
  std::uint32_t packed_cache_fallbacks;
  std::uint32_t packed_cache_stale_refresh;
  // #1179 A3: bumped once per shadow period where the source reached the
  // BASE kRepresentationProbationTouches streak (would have promoted under
  // the pre-A3 rule) but churn_strikes required more evidence, so promotion
  // was withheld and the source stayed on canonical evaluation longer. The
  // direct falsifiable signature of "an economically losing refresh remains
  // canonical" -- zero on a stable-only fixture, nonzero exactly when a
  // fixture forces repeated churn.
  std::uint32_t churn_promotion_deferred;
  // #1179 A3: bumped each time active_stable_touches earns back one
  // churn_strikes credit -- the decay half of the backoff, so a churner that
  // stabilizes is not penalized forever.
  std::uint32_t churn_strikes_recovered;
  // #1235: bumped once per event actually resolved via the packed cache
  // (whether or not it found a conducting winner) -- the counterpart to
  // packed_cache_fallbacks, letting a receipt distinguish "cache present but
  // stale this event" from "cache consulted and current."
  std::uint32_t packed_cache_consult_hits;
  std::uint32_t tensor_rejections;
  std::uint32_t state_owner_migrations;
  std::uint32_t state_owner_rollbacks;
  std::uint32_t state_owner_reclaims;
  std::uint32_t state_owner_shatters;
  std::uint32_t state_owner_lesions;
  std::uint32_t eligibility_rebinds;
};
static_assert(std::is_trivial_v<DirectRepresentationCounters> &&
              std::is_standard_layout_v<DirectRepresentationCounters>);

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_ABI_CUH
