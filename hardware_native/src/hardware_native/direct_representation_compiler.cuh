#ifndef HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_CUH
#define HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_CUH

// #1179 resident representation compiler -- runtime + public API.
//
// Owns two independent, device-resident lifecycles that both feed the one
// #1176 eligibility ledger (see direct_exact_eligibility_device.cuh):
//
//   A. per-source packed-cache promotion (DirectSourceRepresentationState +
//      a small fixed resident reserve of DirectPackedEntry per source).
//      Canonical DirectRoute backing never moves; this is a speed-only
//      execution view with two independent fail-closed staleness guards
//      (source_revision, route_generation).
//
//   B. per-logical-interaction state ownership (DirectRepresentationStateOwner
//      table). The canonical backing itself moves from an explicit route
//      slot into procedural storage; live #1176 records are rebound to the
//      new locator without changing their logical identity or ticket.
//
// Everything here runs device-side from the current tick's frontier/topology
// state; no D2H readback decides representation state on the hot path.

#include "hardware_native/direct_exact_eligibility_device.cuh"

namespace substrate::direct_adult {

struct DirectRepresentationRuntime {
  DirectRepresentationDeviceView view;
  DirectPackedEntry* resident_entries;         // [node_count * kRepresentationResidentReserve]
  std::uint32_t* resident_entry_count;         // [node_count]
  std::uint32_t node_count;
  std::uint32_t state_owner_capacity;          // power of two
};

DirectRepresentationRuntime* create_direct_representation_runtime(
    std::uint32_t node_count, std::uint32_t state_owner_capacity_hint);
void destroy_direct_representation_runtime(DirectRepresentationRuntime* rep);

// Runs the full #1179 per-tick pass: touched-work arbitration over the
// current frontier, packed-cache lifecycle advancement (and resident-cache
// evaluation for any source already `active`), and state-owner migration/
// contradiction handling for materialized #1187 interactions. Called once
// per adult tick from launch_direct_adult_step, after this tick's topology
// epoch(s) have committed and before the frontier/eligibility bank swap.
// block_size is exposed (default 128, matching launch_direct_topology_epoch's
// own convention) so cuda_direct_representation_compiler_contract can prove
// CUDA scheduling does not decide representation-lifecycle outcomes: the
// same frontier processed at two different launch widths must produce the
// same direct_representation_state_root.
void launch_direct_representation_compiler_step(DirectAdultRuntime* runtime,
                                                std::uint32_t frontier_work,
                                                std::uint32_t block_size = 128u);

// Level B (state-owning implicit) half of the per-tick pass, called by
// launch_direct_representation_compiler_step. Declared here (rather than
// kept file-local) only because it is implemented in a separate translation
// unit (direct_representation_state_owner.cu) from the Level A driver above;
// not part of the intended external API.
//
// The retract-retry table kernel this drives is NOT frontier-gated -- it
// idempotently retries every tick regardless of touch activity, including a
// pure-idle tick with frontier_work == 0 (see direct_representation_state_
// owner.cu's own header comment). eligibility_bank/eligibility_count name
// which bank the rebind sub-pass targets: the caller's normal (post-topology-
// epoch, pre-swap) tick passes next_eligibility_bank/next_eligibility_count,
// same as always; a caller invoking this on an otherwise-fully-idle tick that
// never reaches the swap (launch_direct_adult_step's topology_work == 0 fast
// path) must pass the CURRENT eligibility_bank/eligibility_count instead,
// since no swap happens on that path and the "next" bank is stale/unused.
void launch_direct_state_owner_step(DirectAdultRuntime* runtime, std::uint32_t frontier_work,
                                    DirectEligibilityRecord* eligibility_bank,
                                    std::uint32_t* eligibility_count,
                                    std::uint32_t block_size = 128u);

// Focal lesion / unlesion. Both are host-callable, single-source/single-
// interaction operations -- never a whole-brain scan.
bool lesion_representation_packed_source(DirectAdultRuntime* runtime, std::uint32_t source);
bool unlesion_representation_packed_source(DirectAdultRuntime* runtime, std::uint32_t source);
bool lesion_representation_state_owner(DirectAdultRuntime* runtime,
                                       DirectLogicalInteractionId logical_id);
bool unlesion_representation_state_owner(DirectAdultRuntime* runtime,
                                         DirectLogicalInteractionId logical_id);

// Deterministic fingerprint over #1179's own device state, combined into
// DirectAdultCheckpointV0::learned_state_root by direct_adult_checkpoint.cu
// (direct_brain_state_root itself is left untouched -- see that file).
Root256 direct_representation_state_root(const DirectRepresentationRuntime& rep);

// Direct, standalone GPU-event-timeable probes used by
// cuda_direct_representation_compiler_contract to compare the resident
// packed cache against the canonical per-event sparse reference scan on an
// identical workload, with no D2H readback inside the timed region. Each of
// kRepresentationProbeLanes concurrent lanes repeats its evaluation `steps`
// times sequentially (the literal "2048 identical steps" the hard contract
// measures); `*out_checksum` accumulates kRepresentationProbeLanes copies of
// the per-step winning target sum, not one -- callers comparing against an
// expected causal-output value must multiply by kRepresentationProbeLanes.
constexpr std::uint32_t kRepresentationProbeLanes = 256u;
void launch_resident_cache_probe(const DirectAdultRuntime& runtime, std::uint32_t source,
                                 std::uint32_t steps, std::uint32_t* out_checksum);
void launch_canonical_sparse_probe(const DirectAdultRuntime& runtime, std::uint32_t source,
                                   std::uint32_t steps, std::uint32_t* out_checksum);

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_REPRESENTATION_COMPILER_CUH
