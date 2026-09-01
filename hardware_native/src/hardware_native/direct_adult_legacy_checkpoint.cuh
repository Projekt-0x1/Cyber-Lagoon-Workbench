#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH

#include <cstdint>
#include <vector>

#include "hardware_native/direct_causal_action_bridge.hpp"
#include "hardware_native/direct_execution_fabric_abi.cuh"
#include "hardware_native/direct_adult_legacy_oracle.cuh"

namespace substrate::direct_adult {

struct DirectAdultCheckpointV0 {
  Root256 genome_root{};
  Root256 body_root{};
  Root256 birth_root{};
  Root256 learned_state_root{};
  std::uint32_t node_count = 0u;
  std::uint32_t route_count = 0u;
  std::uint32_t route_capacity = 0u;
  std::uint32_t live_route_count = 0u;
  std::uint32_t context_index_capacity = 0u;
  std::uint32_t territory_count = 0u;
  std::uint32_t recurrent_route_count = 0u;
  std::uint32_t long_tract_count = 0u;
  std::uint64_t logical_route_count = 0u;
  std::uint64_t virtual_route_count = 0u;
  std::uint64_t topology_epoch = 0u;
  std::uint32_t topology_free_count = 0u;
  std::uint32_t implicit_family_count = 0u;
  std::uint32_t implicit_exception_capacity = 0u;
  std::uint32_t implicit_exception_count = 0u;
  std::uint32_t tick = 0u;
  std::uint32_t frontier_capacity = 0u;
  std::uint32_t motor_capacity = 0u;
  std::uint32_t eligibility_capacity = 0u;
  std::uint32_t eligibility_bucket_count = 0u;
  std::uint32_t max_tract_delay = 0u;
  ResidentDevelopmentState development{};
  ResidentContextState context_state{};
  DirectResourceEcologyState resource_ecology{};
  std::vector<DirectNode> nodes;
  std::vector<DirectRoute> routes;
  std::vector<ContextRouteIndexEntry> context_index;
  std::vector<DirectRouteSlotMeta> route_slot_meta;
  std::vector<std::uint32_t> topology_free_slots;
  std::vector<std::uint32_t> incoming_degree;
  std::vector<DirectImplicitFamily> implicit_families;
  std::vector<DirectImplicitException> implicit_exceptions;
  std::vector<DirectRetentionState> retention_bank;
  std::vector<DirectMinimalRetentionState> minimal_retention_bank;
  std::vector<ActivityEvent> frontier;
  // Parallel exact-entry authority.  A checkpoint with live frontier data but
  // no matching sidecar is rejected on restore; authority is never inferred
  // from public ActivityEvent fields.
  std::vector<DirectIngressAuthority> frontier_authority;
  std::vector<DirectEligibilityRecord> eligibility_bank;
  std::vector<std::uint32_t> eligibility_bucket_heads;
  std::vector<DirectExecutionMembership> node_memberships;
  std::vector<DirectExecutionMembership> route_memberships;
  std::vector<DirectPackedSparsePanel> packed_panels;
  std::vector<DirectPackedSourceMeta> packed_source_meta;
  std::vector<DirectPackedEntry> packed_entries;
  std::vector<DirectDenseTile> dense_tiles;
  std::vector<DirectTractLane> tract_lanes;
  std::vector<DirectTractPacket> tract_ring_packets;
  std::vector<std::uint32_t> tract_bucket_counts;

  // #1184 checkpoint-pending-I/O: the causal action bridge's own
  // outstanding-ticket ancestry at capture time, plus any adapter returns
  // it had already received but not yet flushed into the membrane. Empty
  // when `capture_direct_adult_checkpoint` was not given a bridge to
  // snapshot. See direct_causal_action_bridge.hpp for the persist-not-
  // cancel policy this implements.
  std::vector<PendingActionTicket> bridge_outstanding_tickets;
  std::vector<BridgeReturn> bridge_completed_returns;

  // #1179 representation state: per-source packed-cache lifecycle, the
  // resident cache reserve, and the state-owner table. All future-affecting
  // adult state, so it is captured/restored exactly like everything else
  // above and folds into learned_state_root (see direct_adult_checkpoint.cu).
  // The touched-work claim scratch (DirectRepresentationDeviceView::claim_ordinal)
  // is intentionally excluded -- it is per-tick arbitration scratch with no
  // meaning across a restart boundary.
  std::uint32_t representation_state_owner_capacity = 0u;
  std::vector<DirectSourceRepresentationState> representation_source_state;
  std::vector<DirectPackedEntry> representation_resident_entries;
  std::vector<std::uint32_t> representation_resident_entry_count;
  std::vector<DirectRepresentationStateOwner> representation_state_owners;
};

// `bridge`, if non-null, has its pending ticket table snapshotted into the
// returned checkpoint (`bridge_outstanding_tickets`/`bridge_completed_returns`).
// Passing nullptr (the default) preserves this function's original
// device-only checkpoint behavior for callers that do not use the bridge.
DirectAdultCheckpointV0 capture_direct_adult_checkpoint(
    const DirectAdultRuntime& runtime, const DirectCausalActionBridge* bridge = nullptr);

// `bridge_out`, if non-null, must point at a freshly constructed (empty)
// bridge; it is rehydrated with `checkpoint`'s bridge_outstanding_tickets/
// bridge_completed_returns before this function returns. Passing nullptr
// (the default) preserves this function's original device-only restore
// behavior.
DirectAdultRuntime restore_direct_adult_checkpoint(const DirectAdultCheckpointV0& checkpoint,
                                                   DirectCausalActionBridge* bridge_out = nullptr);

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH
