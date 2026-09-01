// Patch 0004 of the FULL CUDA network-recipe patch program: the CUDA Life
// Function graph -- device types shared by the construction kernels
// (bcc32_network_construction_kernels.cu) that execute one developmental
// tick: evaluate Gamma's rules against the current sparse frontier, reduce
// conflicting proposals with commutative-only atomics so the result is
// independent of thread scheduling order, and commit the winners into the
// NetworkNode/matter representation (patch 0003).
//
// Scope of this landing (honest, not the full patch program spec yet --
// see the diary entry for the exact gap): the tick evaluates only the
// single-parent rule family (extend/branch/repair); a two-distinct-parent
// fusion proposal is detected (parent_count > 1) and correctly rejected
// rather than silently merged, but does not yet install a connector -- that
// is this patch's own documented next rung, not deferred to a later patch
// number. Tick sequencing is a host-orchestrated loop over these kernels,
// not yet the CUDA-Graph-captured, device-tail-launched single graph the
// patch program specifies (patch program section "Device launch") -- also
// a documented next rung.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_LIFE_FUNCTION_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_LIFE_FUNCTION_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/bcc32_network_matter.cuh"
#include "hardware_native/bcc32_network_page_directory.cuh"
#include "hardware_native/bcc32_network_recipe.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BCC32_LIFE_FUNCTION_HD __host__ __device__
#else
#define BCC32_LIFE_FUNCTION_HD
#endif

namespace substrate::bcc32::network_recipe {

inline constexpr std::uint32_t kInvalidCoordinateWord = 0xffffffffu;

// A currently-active construction source: which live node it is, and where
// it physically sits. This is per-tick frontier bookkeeping (the O_t=(S_t,
// Q_t,B_t) model's Q_t), not part of NetworkNode's own permanent record --
// a node's coordinate is where its *frontier entry* says it last was, not a
// field stored on the node itself.
struct FrontierEntry {
  std::uint32_t node_index;
  std::uint32_t coordinate[3];
};
static_assert(std::is_standard_layout_v<FrontierEntry> && std::is_trivial_v<FrontierEntry>,
              "FrontierEntry must be a fixed-width POD for device residency");

// Bounded per-tick claim table. Every thread proposing construction at the
// same target coordinate this tick must reduce into the same slot (looked
// up by a pure hash of the coordinate) using only commutative atomics
// (min/max/add/or/and/xor), so the final slot contents are identical
// regardless of thread scheduling order -- this is the patch program's
// "Deterministic conflict law". A bounded hash table trades exactness for
// fixed memory (no whole-morphology scan, no per-tick allocation): two
// *different* coordinates that happen to hash to the same slot in the same
// tick are a real, documented approximation limitation -- the second
// writer's z-coordinate silently overwrites the first's non-atomic
// owner_coordinate[2] field, so such a collision can commit a node at the
// wrong z. This is caught by neither this landing's contract nor its
// commit kernel; growing the table or adding open addressing is this
// patch's own next rung, not silently claimed as solved here.
struct TargetClaim {
  std::uint32_t owner_coordinate[3];
  std::uint32_t parent_min;
  std::uint32_t parent_max;
  std::uint32_t parent_count;
  std::uint32_t operation_mask;
  SiteWord chemistry_or;
  SiteWord chemistry_and;
  std::uint32_t lineage_xor;
};
static_assert(std::is_standard_layout_v<TargetClaim> && std::is_trivial_v<TargetClaim>,
              "TargetClaim must be a fixed-width POD for device residency");

inline constexpr std::uint32_t kClaimTableSize = 4096;

enum class CommitOutcome : std::uint32_t {
  kNone = 0,
  kCommitted,
  kTooManyParents,
  kChildSlotsExhausted,
  kPageFull,
  kMatterExhausted,
};

// Device-resident pointers for one developmental tick. Every pointer is
// caller-owned device memory; this struct is passed by value into each
// kernel launch (it is itself small and POD) so no additional host<->device
// copy is needed to hand kernels their working set.
struct LifeFunctionDeviceState {
  const Genome* genome;
  NetworkNode* nodes;
  std::uint32_t node_capacity;
  PageDirectoryEntry* pages;
  NetworkMatterAccount* account;
  TargetClaim* claims;
  FrontierEntry* frontier;
  std::uint32_t* frontier_count;
  // Exact host scheduling mirror of *frontier_count. Device-produced tick
  // reports advance it; it changes launch extent, never developmental law.
  std::uint32_t frontier_size;
  FrontierEntry* next_frontier;
  std::uint32_t* next_frontier_count;
  std::uint32_t tick;
};

BCC32_LIFE_FUNCTION_HD inline std::uint32_t hash_coordinate(const std::uint32_t (&coordinate)[3]) {
  std::uint32_t h = 2166136261u;
  h = (h ^ coordinate[0]) * 16777619u;
  h = (h ^ coordinate[1]) * 16777619u;
  h = (h ^ coordinate[2]) * 16777619u;
  return h % kClaimTableSize;
}

// direction_mode selects one of 6 axis-aligned unit directions; extent is
// the number of sites to step. This is a deliberately simple, deterministic
// construction-front geometry for this landing -- more direction encodings
// (diagonal, field-gradient-following) are a later rung, not a doctrine
// requirement this patch skips (the patch program does not specify an exact
// geometric law, only that construction-front placement/timing be
// explicit and deterministic, which this satisfies).
BCC32_LIFE_FUNCTION_HD inline void step_coordinate(std::uint32_t (&coordinate)[3],
                                                     std::uint32_t direction_mode,
                                                     std::uint32_t extent) {
  switch (direction_mode % 6) {
    case 0: coordinate[0] += extent; break;
    case 1: coordinate[0] -= extent; break;
    case 2: coordinate[1] += extent; break;
    case 3: coordinate[1] -= extent; break;
    case 4: coordinate[2] += extent; break;
    default: coordinate[2] -= extent; break;
  }
}

// Per-tick outcome tally. Commit workers accumulate this in caller-owned
// device memory and the host receives one bounded observer/scheduling copy.
//
// `unsupported_opcode` is not a commit outcome: it is raised in
// evaluate_and_claim_kernel, before any claim exists. It counts, per tick,
// every (frontier source, rule) pair where Gamma's rule was *fully
// eligible* -- inside its tick window and matching the source node's
// chemistry -- and was nonetheless dropped because this Life Function
// version does not implement that rule's opcode (fuse, retract, mature,
// long_tract, endogenous_source). Before this counter existed the drop was
// an unconditional `continue` with no trace in any receipt, so a genome
// specifying development this executor cannot perform produced a
// plausible-looking artifact that silently omitted part of its own recipe.
// A nonzero value means exactly that: the constructed morphology is NOT the
// morphology the genome specifies, and any certification of it certifies a
// truncated recipe. Treat it as a refusal, not a statistic.
struct TickReport {
  std::uint32_t committed;
  std::uint32_t too_many_parents;
  std::uint32_t child_slots_exhausted;
  std::uint32_t page_full;
  std::uint32_t matter_exhausted;
  std::uint32_t new_frontier_size;
  std::uint32_t unsupported_opcode;
};
static_assert(sizeof(TickReport) == 7 * sizeof(std::uint32_t),
              "TickReport must remain one bounded seven-counter copy");
static_assert(std::is_standard_layout_v<TickReport> && std::is_trivial_v<TickReport>,
              "TickReport must be a fixed-width POD for device residency");

// Runs exactly one developmental tick: clear the next-frontier counter (and,
// on the gestation's first tick only, the claim table -- see below), evaluate
// every current frontier source against Gamma's rules (writing
// commutative-reduced proposals into the claim table), commit the winning
// claims into the node/matter/page-directory representation, then swap
// state's frontier <-> next_frontier buffers (and their counts) so the
// caller's next call to run_one_tick() picks up exactly the newly
// constructed nodes. The device report advances frontier_size, eliminating
// the pre-launch frontier-count fence. Advances state.tick by one.
//
// The claim table is only explicitly cleared before tick 0:
// commit_claims_kernel (bcc32_network_construction_kernels.cu) resets every
// slot it processes back to its default/unclaimed state immediately after
// reading it, so the table is already clean for the next tick's
// evaluate_and_claim_kernel writes. Commit workers tally outcomes directly
// into device_report; no kClaimTableSize-wide outcome buffer or reduction
// pass exists. Both changes remove fixed per-tick cost that used to scale
// with allocated claim-table capacity regardless of how small the live
// frontier was that tick (GitHub #1167).
//
// `device_report` must be caller-owned device memory for one TickReport,
// allocated once outside the hot loop.
//
// This is a host-orchestrated sequence of kernel launches on the default
// stream. run_captured_ticks() below is the CUDA-Graph-captured,
// multi-tick-per-launch alternative that removes the remaining per-tick
// host round-trip; see its own comment for the grid-sizing tradeoff that
// entails.
TickReport run_one_tick(LifeFunctionDeviceState& state, TickReport* device_report,
                         std::uint32_t block_size = 128);

// Aggregate outcome tally across every tick a single run_captured_ticks()
// call executes -- the multi-tick analogue of TickReport.
struct CapturedTickReport {
  std::uint32_t committed_total;
  std::uint32_t too_many_parents_total;
  std::uint32_t child_slots_exhausted_total;
  std::uint32_t page_full_total;
  std::uint32_t matter_exhausted_total;
  std::uint32_t final_frontier_size;
  // Summed across every captured tick; see TickReport::unsupported_opcode.
  // Nonzero means this multi-tick run silently skipped genome-specified
  // development and its artifact must not be certified as that genome's.
  std::uint32_t unsupported_opcode_total;
};

// This is run_one_tick()'s own previously-documented next rung, landed:
// captures `tick_count` consecutive developmental ticks into a single CUDA
// graph and replays it once, so growing a construction front to depth N
// costs one host->device graph submission instead of N host-orchestrated
// kernel-launch round-trips. "Developmental semantics does not mean
// developmental slowness" -- a captured, replayed graph is the concrete
// form that takes here.
//
// Unlike run_one_tick(), evaluate_and_claim_kernel is launched every tick
// with a grid sized for the full node_capacity, not the tick's actual
// frontier size -- capture cannot synchronously read a device-resident
// frontier count back to the host mid-capture to pick a tighter grid size.
// The kernel's own internal `i >= *state.frontier_count` bounds check
// keeps this correct; it only launches more idle threads than
// run_one_tick()'s dynamically-sized grid would. This is a real, disclosed
// cost of removing the host round-trip, not a hidden one.
//
// `device_report` has the same caller-owned-once-outside-the-loop contract
// as run_one_tick(). Commit workers accumulate into it directly; the graph
// does not launch a claim-table outcome-reduction pass after every tick. On
// return, `state` is left exactly as `tick_count` calls to run_one_tick()
// would have left it (frontier/next_frontier/tick), so callers may freely
// interleave both paths on the same state.
CapturedTickReport run_captured_ticks(LifeFunctionDeviceState& state, TickReport* device_report,
                                        std::uint32_t tick_count, std::uint32_t block_size = 128);

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_LIFE_FUNCTION_CUH
