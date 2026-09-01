// Patch 0004: the actual CUDA kernels for one developmental tick. See
// bcc32_network_life_function.cuh for the shared types and this landing's
// documented scope (single-parent extend/branch/repair only; fusion is
// detected and rejected, not yet installed).
//
// The five unimplemented opcodes (fuse, retract, mature, long_tract,
// endogenous_source) are still skipped, but the skip is no longer silent:
// evaluate_and_claim_kernel raises TickReport::unsupported_opcode for every
// eligible rule it drops. None of the five has a construction-semantics
// specification anywhere in the repository -- they are declared in
// RuleOpcode (bcc32_network_recipe.hpp), listed in the genome schema, and
// range-checked by the ABI validator, but no document, comment, or test
// states what any of them constructs. Implementing one therefore means
// authoring developmental law, not implementing a spec; the counter exists
// so a genome that needs that law is visibly refused in the meantime.

#include "hardware_native/bcc32_network_construction_kernels.cuh"

namespace substrate::bcc32::network_recipe {

namespace {

__device__ void record_outcome(TickReport& report, CommitOutcome outcome) {
  switch (outcome) {
    case CommitOutcome::kNone: break;
    case CommitOutcome::kCommitted: atomicAdd(&report.committed, 1u); break;
    case CommitOutcome::kTooManyParents: atomicAdd(&report.too_many_parents, 1u); break;
    case CommitOutcome::kChildSlotsExhausted:
      atomicAdd(&report.child_slots_exhausted, 1u);
      break;
    case CommitOutcome::kPageFull: atomicAdd(&report.page_full, 1u); break;
    case CommitOutcome::kMatterExhausted: atomicAdd(&report.matter_exhausted, 1u); break;
  }
}

}  // namespace

__global__ void clear_claims_kernel(TargetClaim* claims) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kClaimTableSize) return;
  claims[i].owner_coordinate[0] = kInvalidCoordinateWord;
  claims[i].owner_coordinate[1] = kInvalidCoordinateWord;
  claims[i].owner_coordinate[2] = kInvalidCoordinateWord;
  claims[i].parent_min = 0xffffffffu;
  claims[i].parent_max = 0u;
  claims[i].parent_count = 0u;
  claims[i].operation_mask = 0u;
  claims[i].chemistry_or = 0u;
  claims[i].chemistry_and = 0xffffffffu;
  claims[i].lineage_xor = 0u;
}

__global__ void clear_next_frontier_kernel(std::uint32_t* next_frontier_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *next_frontier_count = 0u;
}

__global__ void evaluate_and_claim_kernel(LifeFunctionDeviceState state, TickReport* report) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *state.frontier_count) return;

  const FrontierEntry source = state.frontier[i];
  const NetworkNode& node = state.nodes[source.node_index];

  // Accumulated per thread and flushed once after the loop, so a genome with
  // many unimplemented rules costs one atomic per frontier source instead of
  // one per (source, rule) pair. Addition is commutative and every
  // contributing thread is selected by the frontier bounds check above, so
  // the flushed total is identical for any block size or launch geometry.
  std::uint32_t unsupported_here = 0u;

  for (std::uint32_t r = 0; r < state.genome->header.rule_count; ++r) {
    const ConstructionRule& rule = state.genome->rules[r];
    if (state.tick < rule.begin_tick || state.tick >= rule.end_tick) continue;
    if ((node.chemistry & rule.require_mask) != rule.require_value) continue;
    if (rule.opcode != RuleOpcode::extend && rule.opcode != RuleOpcode::branch &&
        rule.opcode != RuleOpcode::repair) {
      // fuse/retract/mature/long_tract/endogenous_source are not implemented
      // by this Life Function version. This rule passed both eligibility
      // tests and is being dropped anyway, which changes what gets
      // constructed. Record it so the tick report shows the genome was only
      // partially executed rather than handing the caller a silently
      // truncated recipe.
      ++unsupported_here;
      continue;
    }

    std::uint32_t target[3] = {source.coordinate[0], source.coordinate[1], source.coordinate[2]};
    step_coordinate(target, rule.direction_mode, rule.extent);

    const std::uint32_t slot = hash_coordinate(target);
    TargetClaim& claim = state.claims[slot];

    auto* owner01 = reinterpret_cast<unsigned long long*>(&claim.owner_coordinate[0]);
    const unsigned long long unclaimed =
        (static_cast<unsigned long long>(kInvalidCoordinateWord) << 32) | kInvalidCoordinateWord;
    const unsigned long long desired =
        (static_cast<unsigned long long>(target[1]) << 32) | target[0];
    const unsigned long long observed = atomicCAS(owner01, unclaimed, desired);
    const bool established_or_matches =
        observed == unclaimed || (static_cast<std::uint32_t>(observed) == target[0] &&
                                   static_cast<std::uint32_t>(observed >> 32) == target[1]);
    if (established_or_matches) {
      claim.owner_coordinate[2] = target[2];
    }

    atomicMin(&claim.parent_min, source.node_index);
    atomicMax(&claim.parent_max, source.node_index);
    atomicAdd(&claim.parent_count, 1u);
    atomicOr(&claim.operation_mask, 1u << static_cast<std::uint32_t>(rule.opcode));
    atomicOr(&claim.chemistry_or, rule.write_value);
    atomicAnd(&claim.chemistry_and, rule.write_value);
    atomicXor(&claim.lineage_xor, node.lineage);
    break;  // first matching rule only -- deterministic, fixed array order
  }

  if (unsupported_here != 0u) {
    atomicAdd(&report->unsupported_opcode, unsupported_here);
  }
}

__global__ void commit_claims_kernel(LifeFunctionDeviceState state, TickReport* report) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kClaimTableSize) return;

  TargetClaim& claim = state.claims[i];

  // Snapshot this tick's claim contents, then immediately reset the slot to
  // its default/unclaimed state -- every branch below reads only these
  // locals, never `claim` again, so the reset is unconditional regardless
  // of which outcome this slot resolves to (including the untouched-slot
  // kNone case, where resetting already-default fields is a no-op). This
  // folds the per-tick claim-table clear into commit (GitHub #1167's
  // comment, item 2) instead of paying a separate full-kClaimTableSize-wide
  // clear_claims_kernel sweep every tick regardless of how small the live
  // frontier is. clear_claims_kernel is still required exactly once, before
  // a gestation's first tick, since nothing has committed yet to have reset
  // a fresh allocation's contents.
  const std::uint32_t owner0 = claim.owner_coordinate[0];
  const std::uint32_t owner1 = claim.owner_coordinate[1];
  const std::uint32_t owner2 = claim.owner_coordinate[2];
  const std::uint32_t parent_min = claim.parent_min;
  const std::uint32_t parent_count = claim.parent_count;
  const SiteWord chemistry_or = claim.chemistry_or;
  const std::uint32_t lineage_xor = claim.lineage_xor;

  claim.owner_coordinate[0] = kInvalidCoordinateWord;
  claim.owner_coordinate[1] = kInvalidCoordinateWord;
  claim.owner_coordinate[2] = kInvalidCoordinateWord;
  claim.parent_min = 0xffffffffu;
  claim.parent_max = 0u;
  claim.parent_count = 0u;
  claim.operation_mask = 0u;
  claim.chemistry_or = 0u;
  claim.chemistry_and = 0xffffffffu;
  claim.lineage_xor = 0u;

  if (parent_count == 0 || owner0 == kInvalidCoordinateWord) {
    return;
  }

  if (parent_count > 1) {
    // A real two-distinct-lineage fusion needs to verify the parents'
    // lineages actually differ and install a connector edge to both --
    // deferred to this patch's own next rung. Rejecting outright here is
    // the honest choice: silently picking one contributing parent would be
    // exactly the "traversal/order chooses the outcome" forbidden partial.
    record_outcome(*report, CommitOutcome::kTooManyParents);
    return;
  }

  const std::uint32_t parent_index = parent_min;  // == parent_max, single contributor
  NetworkNode& parent = state.nodes[parent_index];

  std::uint32_t free_child_slot = kMaximumChildren;
  for (std::uint32_t c = 0; c < kMaximumChildren; ++c) {
    if (parent.child[c] == kInvalidNodeIndex) {
      free_child_slot = c;
      break;
    }
  }
  if (free_child_slot == kMaximumChildren) {
    record_outcome(*report, CommitOutcome::kChildSlotsExhausted);
    return;
  }

  const std::uint32_t target[3] = {owner0, owner1, owner2};
  const std::uint32_t page_id = page_index_for_coordinate(target);
  std::uint32_t slot_in_page = 0;
  if (reserve_node_slot(state.pages[page_id], slot_in_page) != PageReservationError::kNone) {
    record_outcome(*report, CommitOutcome::kPageFull);
    return;
  }

  if (debit_matter(*state.account, MatterBucket::kLiveNode, 1) != MatterCommitError::kNone) {
    record_outcome(*report, CommitOutcome::kMatterExhausted);
    return;
  }

  const std::uint32_t global_index = global_node_index(page_id, slot_in_page);
  if (global_index >= state.node_capacity) {
    // This landing's bounded test allocation is smaller than the full
    // page-directory address space (kMaxNetworkNodes); a real out-of-bound
    // global index for a bounded allocation is a genuine capacity fault,
    // not a silent drop -- the matter already debited above stays debited
    // (this candidate consumed real budget attempting construction that
    // didn't fit; that is honest accounting, not a leak, since debited
    // matter is never fabricated back).
    record_outcome(*report, CommitOutcome::kMatterExhausted);
    return;
  }

  // lineage_xor of exactly one contributor equals that contributor's own
  // lineage -- the parent's lineage is inherited by construction, matching
  // "extend/branch/repair" all being same-lineage operations.
  state.nodes[global_index] = empty_network_node(chemistry_or, lineage_xor, state.tick);
  state.nodes[global_index].parent[0] = parent_index;
  state.nodes[global_index].parent[1] = kInvalidNodeIndex;

  parent.child[free_child_slot] = global_index;
  parent.edge_chemistry[free_child_slot] = chemistry_or;

  const std::uint32_t next_i = atomicAdd(state.next_frontier_count, 1u);
  if (next_i < state.node_capacity) {
    state.next_frontier[next_i].node_index = global_index;
    state.next_frontier[next_i].coordinate[0] = target[0];
    state.next_frontier[next_i].coordinate[1] = target[1];
    state.next_frontier[next_i].coordinate[2] = target[2];
    atomicMax(&report->new_frontier_size, next_i + 1u);
  }

  record_outcome(*report, CommitOutcome::kCommitted);
}

}  // namespace substrate::bcc32::network_recipe
