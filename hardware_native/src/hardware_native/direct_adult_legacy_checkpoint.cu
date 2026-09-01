#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "hardware_native/direct_adult_legacy_checkpoint.cuh"
#include "hardware_native/direct_dynamic_topology_arena.cuh"
#include "hardware_native/direct_execution_fabric.cuh"
#include "hardware_native/direct_implicit_causal_mesh.cuh"
#include "hardware_native/direct_resource_ecology_legacy.cuh"
#include "hardware_native/direct_retention_policy.cuh"
#include "hardware_native/direct_representation_compiler.cuh"
#include "hardware_native/direct_network_recipe_abi.cuh"

namespace substrate::direct_adult {
namespace {

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

}  // namespace

DirectAdultCheckpointV0 capture_direct_adult_checkpoint(const DirectAdultRuntime& runtime,
                                                        const DirectCausalActionBridge* bridge) {
  DirectAdultCheckpointV0 checkpoint{};
  checkpoint.genome_root = runtime.brain.genome_root;
  checkpoint.body_root = runtime.brain.body_root;
  checkpoint.birth_root = runtime.brain.birth_root;
  checkpoint.node_count = runtime.brain.node_count;
  checkpoint.route_count = runtime.brain.route_count;
  checkpoint.route_capacity = runtime.brain.route_capacity;
  checkpoint.context_index_capacity = runtime.brain.context_index_capacity;
  checkpoint.territory_count = runtime.brain.territory_count;
  checkpoint.recurrent_route_count = runtime.brain.recurrent_route_count;
  checkpoint.long_tract_count = runtime.brain.long_tract_count;
  checkpoint.logical_route_count = runtime.brain.logical_route_count;
  checkpoint.virtual_route_count = runtime.brain.virtual_route_count;
  checkpoint.tick = runtime.tick;
  checkpoint.frontier_capacity = runtime.frontier_capacity;
  checkpoint.motor_capacity = runtime.motor_capacity;
  checkpoint.eligibility_capacity = runtime.eligibility_capacity;
  checkpoint.eligibility_bucket_count = runtime.eligibility_bucket_count;
  checkpoint.implicit_family_count = runtime.brain.implicit.family_count;
  checkpoint.implicit_exception_capacity = runtime.brain.implicit.exception_capacity;

  checkpoint.nodes.resize(checkpoint.node_count);
  checkpoint.routes.resize(checkpoint.route_capacity);
  checkpoint.context_index.resize(checkpoint.context_index_capacity);
  checkpoint.route_slot_meta.resize(checkpoint.route_capacity);
  checkpoint.topology_free_slots.resize(checkpoint.route_capacity);
  checkpoint.incoming_degree.resize(checkpoint.node_count);
  checkpoint.implicit_families.resize(checkpoint.implicit_family_count);
  checkpoint.implicit_exceptions.resize(checkpoint.implicit_exception_capacity);

  require_cuda(cudaMemcpy(checkpoint.nodes.data(), runtime.brain.nodes,
                          sizeof(DirectNode) * checkpoint.nodes.size(), cudaMemcpyDeviceToHost),
               "checkpoint nodes");
  require_cuda(cudaMemcpy(checkpoint.routes.data(), runtime.brain.routes,
                          sizeof(DirectRoute) * checkpoint.routes.size(), cudaMemcpyDeviceToHost),
               "checkpoint routes");
  require_cuda(cudaMemcpy(checkpoint.context_index.data(), runtime.brain.context_index,
                          sizeof(ContextRouteIndexEntry) * checkpoint.context_index.size(),
                          cudaMemcpyDeviceToHost),
               "checkpoint context route index");
  require_cuda(cudaMemcpy(checkpoint.route_slot_meta.data(), runtime.brain.topology.slot_meta,
                          sizeof(DirectRouteSlotMeta) * checkpoint.route_slot_meta.size(),
                          cudaMemcpyDeviceToHost),
               "checkpoint route slot metadata");
  require_cuda(cudaMemcpy(checkpoint.topology_free_slots.data(), runtime.brain.topology.free_slots,
                          sizeof(std::uint32_t) * checkpoint.topology_free_slots.size(),
                          cudaMemcpyDeviceToHost),
               "checkpoint topology free slots");
  require_cuda(
      cudaMemcpy(checkpoint.incoming_degree.data(), runtime.brain.topology.incoming_degree,
                 sizeof(std::uint32_t) * checkpoint.incoming_degree.size(), cudaMemcpyDeviceToHost),
      "checkpoint topology incoming degree");
  require_cuda(cudaMemcpy(&checkpoint.topology_free_count, runtime.brain.topology.free_count,
                          sizeof(checkpoint.topology_free_count), cudaMemcpyDeviceToHost),
               "checkpoint topology free count");
  require_cuda(cudaMemcpy(&checkpoint.topology_epoch, runtime.brain.topology.epoch,
                          sizeof(checkpoint.topology_epoch), cudaMemcpyDeviceToHost),
               "checkpoint topology epoch");
  require_cuda(cudaMemcpy(&checkpoint.development, runtime.brain.development,
                          sizeof(checkpoint.development), cudaMemcpyDeviceToHost),
               "checkpoint resident development");
  require_cuda(cudaMemcpy(&checkpoint.live_route_count, runtime.brain.live_route_count,
                          sizeof(checkpoint.live_route_count), cudaMemcpyDeviceToHost),
               "checkpoint live route count");
  require_cuda(cudaMemcpy(&checkpoint.context_state, runtime.context_state,
                          sizeof(checkpoint.context_state), cudaMemcpyDeviceToHost),
               "checkpoint resident context");
  if (checkpoint.implicit_family_count != 0u) {
    require_cuda(cudaMemcpy(checkpoint.implicit_families.data(), runtime.brain.implicit.families,
                            sizeof(DirectImplicitFamily) * checkpoint.implicit_families.size(),
                            cudaMemcpyDeviceToHost),
                 "checkpoint implicit families");
  }
  if (checkpoint.implicit_exception_capacity != 0u) {
    require_cuda(
        cudaMemcpy(checkpoint.implicit_exceptions.data(), runtime.brain.implicit.exceptions,
                   sizeof(DirectImplicitException) * checkpoint.implicit_exceptions.size(),
                   cudaMemcpyDeviceToHost),
        "checkpoint implicit exceptions");
    require_cuda(
        cudaMemcpy(&checkpoint.implicit_exception_count, runtime.brain.implicit.exception_count,
                   sizeof(checkpoint.implicit_exception_count), cudaMemcpyDeviceToHost),
        "checkpoint implicit exception count");
  }

  if (runtime.brain.resource_ecology != nullptr) {
    require_cuda(cudaMemcpy(&checkpoint.resource_ecology, runtime.brain.resource_ecology,
                            sizeof(checkpoint.resource_ecology), cudaMemcpyDeviceToHost),
                 "checkpoint resource ecology");
  }
  if (runtime.brain.retention_bank != nullptr) {
    checkpoint.retention_bank.resize(checkpoint.route_capacity);
    require_cuda(cudaMemcpy(checkpoint.retention_bank.data(), runtime.brain.retention_bank,
                            sizeof(DirectRetentionState) * checkpoint.route_capacity, cudaMemcpyDeviceToHost),
                 "checkpoint retention bank");
  }
  if (runtime.brain.minimal_retention_bank != nullptr) {
    checkpoint.minimal_retention_bank.resize(checkpoint.route_capacity);
    require_cuda(cudaMemcpy(checkpoint.minimal_retention_bank.data(), runtime.brain.minimal_retention_bank,
                            sizeof(DirectMinimalRetentionState) * checkpoint.route_capacity, cudaMemcpyDeviceToHost),
                 "checkpoint minimal retention bank");
  }

  std::uint32_t frontier_count = 0u;
  std::uint32_t eligibility_count = 0u;
  require_cuda(cudaMemcpy(&frontier_count, runtime.frontier_count, sizeof(frontier_count),
                          cudaMemcpyDeviceToHost),
               "checkpoint frontier count");
  require_cuda(cudaMemcpy(&eligibility_count, runtime.eligibility_count, sizeof(eligibility_count),
                          cudaMemcpyDeviceToHost),
               "checkpoint eligibility count");
  frontier_count = std::min(frontier_count, runtime.frontier_capacity);
  eligibility_count = std::min(eligibility_count, runtime.eligibility_capacity);
  checkpoint.frontier.resize(frontier_count);
  checkpoint.frontier_authority.resize(frontier_count);
  checkpoint.eligibility_bank.resize(eligibility_count);
  checkpoint.eligibility_bucket_heads.resize(checkpoint.eligibility_bucket_count);
  if (frontier_count != 0u) {
    require_cuda(cudaMemcpy(checkpoint.frontier.data(), runtime.frontier,
                            sizeof(ActivityEvent) * frontier_count, cudaMemcpyDeviceToHost),
                 "checkpoint frontier");
    require_cuda(
        cudaMemcpy(checkpoint.frontier_authority.data(), runtime.frontier_authority,
                   sizeof(DirectIngressAuthority) * frontier_count, cudaMemcpyDeviceToHost),
        "checkpoint frontier authority");
  }
  if (eligibility_count != 0u) {
    require_cuda(
        cudaMemcpy(checkpoint.eligibility_bank.data(), runtime.eligibility_bank,
                   sizeof(DirectEligibilityRecord) * eligibility_count, cudaMemcpyDeviceToHost),
        "checkpoint eligibility bank");
  }
  if (checkpoint.eligibility_bucket_count != 0u) {
    require_cuda(
        cudaMemcpy(checkpoint.eligibility_bucket_heads.data(), runtime.eligibility_bucket_heads,
                   sizeof(std::uint32_t) * checkpoint.eligibility_bucket_count,
                   cudaMemcpyDeviceToHost),
        "checkpoint eligibility index");
  }

  // Checkpoint Execution Fabric State
  if (runtime.fabric != nullptr) {
    const DirectExecutionFabricRuntime& fab = *runtime.fabric;
    checkpoint.max_tract_delay = fab.max_tract_delay;

    checkpoint.node_memberships.resize(checkpoint.node_count);
    require_cuda(cudaMemcpy(checkpoint.node_memberships.data(), fab.view.node_memberships,
                            sizeof(DirectExecutionMembership) * checkpoint.node_count,
                            cudaMemcpyDeviceToHost),
                 "checkpoint node memberships");

    checkpoint.route_memberships.resize(checkpoint.route_capacity);
    require_cuda(cudaMemcpy(checkpoint.route_memberships.data(), fab.view.route_memberships,
                            sizeof(DirectExecutionMembership) * checkpoint.route_capacity,
                            cudaMemcpyDeviceToHost),
                 "checkpoint route memberships");

    if (fab.packed_panel_count != 0u) {
      checkpoint.packed_panels.resize(fab.packed_panel_count);
      require_cuda(cudaMemcpy(checkpoint.packed_panels.data(), fab.view.packed_panels,
                              sizeof(DirectPackedSparsePanel) * fab.packed_panel_count,
                              cudaMemcpyDeviceToHost),
                   "checkpoint packed panels");

      checkpoint.packed_source_meta.resize(checkpoint.node_count);
      require_cuda(cudaMemcpy(checkpoint.packed_source_meta.data(), fab.view.packed_source_meta,
                              sizeof(DirectPackedSourceMeta) * checkpoint.node_count,
                              cudaMemcpyDeviceToHost),
                   "checkpoint packed source meta");

      checkpoint.packed_entries.resize(fab.packed_entry_count);
      if (fab.packed_entry_count != 0u) {
        require_cuda(
            cudaMemcpy(checkpoint.packed_entries.data(), fab.view.packed_entries,
                       sizeof(DirectPackedEntry) * fab.packed_entry_count, cudaMemcpyDeviceToHost),
            "checkpoint packed entries");
      }
    }

    if (fab.dense_tile_count != 0u) {
      checkpoint.dense_tiles.resize(fab.dense_tile_count);
      require_cuda(
          cudaMemcpy(checkpoint.dense_tiles.data(), fab.view.dense_tiles,
                     sizeof(DirectDenseTile) * fab.dense_tile_count, cudaMemcpyDeviceToHost),
          "checkpoint dense tiles");
    }

    if (fab.tract_lane_count != 0u) {
      checkpoint.tract_lanes.resize(fab.tract_lane_count);
      require_cuda(
          cudaMemcpy(checkpoint.tract_lanes.data(), fab.view.tract_lanes,
                     sizeof(DirectTractLane) * fab.tract_lane_count, cudaMemcpyDeviceToHost),
          "checkpoint tract lanes");
    }

    const std::uint32_t ring_buckets = fab.max_tract_delay + 1u;
    checkpoint.tract_bucket_counts.resize(ring_buckets);
    require_cuda(cudaMemcpy(checkpoint.tract_bucket_counts.data(), fab.view.tract_bucket_counts,
                            sizeof(std::uint32_t) * ring_buckets, cudaMemcpyDeviceToHost),
                 "checkpoint tract bucket counts");

    const std::uint32_t stride = runtime.brain.node_count;
    const std::uint32_t ring_total = ring_buckets * stride;
    checkpoint.tract_ring_packets.resize(ring_total);
    require_cuda(cudaMemcpy(checkpoint.tract_ring_packets.data(), fab.view.tract_ring_packets,
                            sizeof(DirectTractPacket) * ring_total, cudaMemcpyDeviceToHost),
                 "checkpoint tract ring packets");
  }
  // #1179 representation state.
  if (runtime.representation != nullptr) {
    const DirectRepresentationRuntime& rep = *runtime.representation;
    checkpoint.representation_state_owner_capacity = rep.state_owner_capacity;
    checkpoint.representation_source_state.resize(rep.node_count);
    require_cuda(cudaMemcpy(checkpoint.representation_source_state.data(), rep.view.source_state,
                            sizeof(DirectSourceRepresentationState) * rep.node_count,
                            cudaMemcpyDeviceToHost),
                 "checkpoint representation source state");
    checkpoint.representation_resident_entries.resize(
        static_cast<std::size_t>(rep.node_count) * kRepresentationResidentReserve);
    require_cuda(cudaMemcpy(checkpoint.representation_resident_entries.data(), rep.resident_entries,
                            sizeof(DirectPackedEntry) * checkpoint.representation_resident_entries.size(),
                            cudaMemcpyDeviceToHost),
                 "checkpoint representation resident entries");
    checkpoint.representation_resident_entry_count.resize(rep.node_count);
    require_cuda(cudaMemcpy(checkpoint.representation_resident_entry_count.data(),
                            rep.resident_entry_count, sizeof(std::uint32_t) * rep.node_count,
                            cudaMemcpyDeviceToHost),
                 "checkpoint representation resident entry counts");
    checkpoint.representation_state_owners.resize(rep.state_owner_capacity);
    require_cuda(cudaMemcpy(checkpoint.representation_state_owners.data(), rep.view.state_owners,
                            sizeof(DirectRepresentationStateOwner) * rep.state_owner_capacity,
                            cudaMemcpyDeviceToHost),
                 "checkpoint representation state owners");
  }

  const Root256 brain_root = direct_brain_state_root(runtime.brain);
  const Root256 representation_root = runtime.representation != nullptr
      ? direct_representation_state_root(*runtime.representation)
      : Root256{};
  struct CombinedRoot { Root256 brain; Root256 representation; } combined_root{brain_root, representation_root};
  checkpoint.learned_state_root =
      direct_network::recipe::content_root(&combined_root, sizeof(combined_root));

  // #1184 checkpoint-pending-I/O: persist the bridge's own outstanding
  // ticket ancestry (never the GPU's job) so an in-flight external
  // transaction survives restart instead of silently vanishing.
  if (bridge != nullptr) {
    const DirectCausalActionBridge::BridgeCheckpointState pending = bridge->capture_pending_state();
    checkpoint.bridge_outstanding_tickets = pending.outstanding;
    checkpoint.bridge_completed_returns = pending.completed;
  }
  return checkpoint;
}

DirectAdultRuntime restore_direct_adult_checkpoint(const DirectAdultCheckpointV0& checkpoint,
                                                   DirectCausalActionBridge* bridge_out) {
  if (checkpoint.frontier_authority.size() != checkpoint.frontier.size()) {
    throw std::invalid_argument("direct adult checkpoint frontier authority shape mismatch");
  }
  if (checkpoint.nodes.size() != checkpoint.node_count ||
      checkpoint.routes.size() != checkpoint.route_capacity ||
      checkpoint.context_index.size() != checkpoint.context_index_capacity ||
      checkpoint.route_slot_meta.size() != checkpoint.route_capacity ||
      checkpoint.topology_free_slots.size() != checkpoint.route_capacity ||
      checkpoint.incoming_degree.size() != checkpoint.node_count ||
      checkpoint.implicit_families.size() != checkpoint.implicit_family_count ||
      checkpoint.implicit_exceptions.size() != checkpoint.implicit_exception_capacity ||
      checkpoint.frontier.size() > checkpoint.frontier_capacity ||
      checkpoint.eligibility_bank.size() > checkpoint.eligibility_capacity ||
      checkpoint.eligibility_bucket_heads.size() != checkpoint.eligibility_bucket_count) {
    throw std::invalid_argument("direct adult checkpoint shape is inconsistent");
  }

  DirectBrainV01 brain{};
  brain.node_count = checkpoint.node_count;
  brain.route_count = checkpoint.route_count;
  brain.route_capacity = checkpoint.route_capacity;
  brain.context_index_capacity = checkpoint.context_index_capacity;
  brain.territory_count = checkpoint.territory_count;
  brain.recurrent_route_count = checkpoint.recurrent_route_count;
  brain.long_tract_count = checkpoint.long_tract_count;
  brain.logical_route_count = checkpoint.logical_route_count;
  brain.virtual_route_count = checkpoint.virtual_route_count;
  brain.genome_root = checkpoint.genome_root;
  brain.body_root = checkpoint.body_root;
  brain.birth_root = checkpoint.birth_root;

  require_cuda(cudaMalloc(&brain.nodes, sizeof(DirectNode) * brain.node_count), "restore nodes");
  require_cuda(cudaMalloc(&brain.routes, sizeof(DirectRoute) * brain.route_capacity),
               "restore routes");
  require_cuda(cudaMalloc(&brain.live_route_count, sizeof(std::uint32_t)),
               "restore live route count");
  require_cuda(cudaMalloc(&brain.context_index,
                          sizeof(ContextRouteIndexEntry) * brain.context_index_capacity),
               "restore context route index");
  require_cuda(cudaMalloc(&brain.development, sizeof(ResidentDevelopmentState)),
               "restore resident development");
  require_cuda(cudaMalloc(&brain.resource_ecology, sizeof(DirectResourceEcologyState)),
               "restore resource ecology");
  require_cuda(cudaMalloc(&brain.retention_bank, sizeof(DirectRetentionState) * brain.route_capacity),
               "restore retention bank");
  require_cuda(cudaMalloc(&brain.minimal_retention_bank, sizeof(DirectMinimalRetentionState) * brain.route_capacity),
               "restore minimal retention bank");

  require_cuda(cudaMemcpy(brain.nodes, checkpoint.nodes.data(),
                          sizeof(DirectNode) * brain.node_count, cudaMemcpyHostToDevice),
               "restore node state");
  require_cuda(cudaMemcpy(brain.routes, checkpoint.routes.data(),
                          sizeof(DirectRoute) * brain.route_capacity, cudaMemcpyHostToDevice),
               "restore route state");
  require_cuda(cudaMemcpy(brain.live_route_count, &checkpoint.live_route_count,
                          sizeof(checkpoint.live_route_count), cudaMemcpyHostToDevice),
               "restore live route count");
  require_cuda(cudaMemcpy(brain.context_index, checkpoint.context_index.data(),
                          sizeof(ContextRouteIndexEntry) * brain.context_index_capacity,
                          cudaMemcpyHostToDevice),
               "restore context route index");
  require_cuda(cudaMemcpy(brain.development, &checkpoint.development,
                          sizeof(checkpoint.development), cudaMemcpyHostToDevice),
               "restore resident development state");
  require_cuda(cudaMemcpy(brain.resource_ecology, &checkpoint.resource_ecology,
                          sizeof(checkpoint.resource_ecology), cudaMemcpyHostToDevice),
               "restore resource ecology state");
  if (!checkpoint.retention_bank.empty()) {
    require_cuda(cudaMemcpy(brain.retention_bank, checkpoint.retention_bank.data(),
                            sizeof(DirectRetentionState) * checkpoint.retention_bank.size(), cudaMemcpyHostToDevice),
                 "restore retention bank");
  } else {
    initialize_direct_retention_bank(brain.retention_bank, brain.route_capacity);
  }
  if (!checkpoint.minimal_retention_bank.empty()) {
    require_cuda(cudaMemcpy(brain.minimal_retention_bank, checkpoint.minimal_retention_bank.data(),
                            sizeof(DirectMinimalRetentionState) * checkpoint.minimal_retention_bank.size(), cudaMemcpyHostToDevice),
                 "restore minimal retention bank");
  } else {
    initialize_direct_minimal_retention_bank(brain.minimal_retention_bank, brain.route_capacity);
  }

  initialize_direct_topology_state(&brain);
  require_cuda(cudaMemcpy(brain.topology.slot_meta, checkpoint.route_slot_meta.data(),
                          sizeof(DirectRouteSlotMeta) * checkpoint.route_slot_meta.size(),
                          cudaMemcpyHostToDevice),
               "restore topology slot metadata");
  require_cuda(cudaMemcpy(brain.topology.free_slots, checkpoint.topology_free_slots.data(),
                          sizeof(std::uint32_t) * checkpoint.topology_free_slots.size(),
                          cudaMemcpyHostToDevice),
               "restore topology free slots");
  require_cuda(
      cudaMemcpy(brain.topology.incoming_degree, checkpoint.incoming_degree.data(),
                 sizeof(std::uint32_t) * checkpoint.incoming_degree.size(), cudaMemcpyHostToDevice),
      "restore topology incoming degree");
  require_cuda(cudaMemcpy(brain.topology.free_count, &checkpoint.topology_free_count,
                          sizeof(checkpoint.topology_free_count), cudaMemcpyHostToDevice),
               "restore topology free count");
  require_cuda(cudaMemcpy(brain.topology.epoch, &checkpoint.topology_epoch,
                          sizeof(checkpoint.topology_epoch), cudaMemcpyHostToDevice),
               "restore topology epoch");

  initialize_direct_implicit_state(&brain, checkpoint.implicit_families.data(),
                                   checkpoint.implicit_family_count, checkpoint.virtual_route_count,
                                   checkpoint.implicit_exception_capacity);
  if (checkpoint.implicit_exception_capacity != 0u) {
    require_cuda(cudaMemcpy(brain.implicit.exceptions, checkpoint.implicit_exceptions.data(),
                            sizeof(DirectImplicitException) * checkpoint.implicit_exceptions.size(),
                            cudaMemcpyHostToDevice),
                 "restore implicit exceptions");
    require_cuda(cudaMemcpy(brain.implicit.exception_count, &checkpoint.implicit_exception_count,
                            sizeof(checkpoint.implicit_exception_count), cudaMemcpyHostToDevice),
                 "restore implicit exception count");
  }

  DirectAdultRuntime runtime =
      create_direct_adult_runtime(brain, checkpoint.frontier_capacity, checkpoint.motor_capacity,
                                  checkpoint.eligibility_capacity);
  if (runtime.eligibility_bucket_count != checkpoint.eligibility_bucket_count) {
    destroy_direct_adult_runtime(&runtime, true);
    throw std::invalid_argument("direct adult checkpoint eligibility index shape differs");
  }
  if (runtime.representation != nullptr &&
      runtime.representation->state_owner_capacity != checkpoint.representation_state_owner_capacity) {
    destroy_direct_adult_runtime(&runtime, true);
    throw std::invalid_argument("direct adult checkpoint representation shape differs");
  }
  runtime.tick = checkpoint.tick;
  require_cuda(cudaMemcpy(runtime.context_state, &checkpoint.context_state,
                          sizeof(checkpoint.context_state), cudaMemcpyHostToDevice),
               "restore resident context");
  const std::uint32_t frontier_count = static_cast<std::uint32_t>(checkpoint.frontier.size());
  const std::uint32_t eligibility_count =
      static_cast<std::uint32_t>(checkpoint.eligibility_bank.size());
  if (frontier_count != 0u) {
    require_cuda(cudaMemcpy(runtime.frontier, checkpoint.frontier.data(),
                            sizeof(ActivityEvent) * frontier_count, cudaMemcpyHostToDevice),
                 "restore live frontier");
    require_cuda(
        cudaMemcpy(runtime.frontier_authority, checkpoint.frontier_authority.data(),
                   sizeof(DirectIngressAuthority) * frontier_count, cudaMemcpyHostToDevice),
        "restore frontier authority");
  }
  if (eligibility_count != 0u) {
    require_cuda(
        cudaMemcpy(runtime.eligibility_bank, checkpoint.eligibility_bank.data(),
                   sizeof(DirectEligibilityRecord) * eligibility_count, cudaMemcpyHostToDevice),
        "restore eligibility bank");
  }
  if (checkpoint.eligibility_bucket_count != 0u) {
    require_cuda(
        cudaMemcpy(runtime.eligibility_bucket_heads, checkpoint.eligibility_bucket_heads.data(),
                   sizeof(std::uint32_t) * checkpoint.eligibility_bucket_count,
                   cudaMemcpyHostToDevice),
        "restore eligibility index");
  }
  require_cuda(cudaMemcpy(runtime.frontier_count, &frontier_count, sizeof(frontier_count),
                          cudaMemcpyHostToDevice),
               "restore frontier count");
  require_cuda(cudaMemcpy(runtime.eligibility_count, &eligibility_count, sizeof(eligibility_count),
                          cudaMemcpyHostToDevice),
               "restore eligibility count");

  // Restore Execution Fabric State & in-flight tract launch bounds
  std::uint32_t pending_tract_total = 0u;
  if (runtime.fabric != nullptr) {
    DirectExecutionFabricRuntime& fab = *runtime.fabric;
    if (!checkpoint.node_memberships.empty()) {
      require_cuda(
          cudaMemcpy(fab.view.node_memberships, checkpoint.node_memberships.data(),
                     sizeof(DirectExecutionMembership) * checkpoint.node_memberships.size(),
                     cudaMemcpyHostToDevice),
          "restore node memberships");
    }
    if (!checkpoint.route_memberships.empty()) {
      require_cuda(
          cudaMemcpy(fab.view.route_memberships, checkpoint.route_memberships.data(),
                     sizeof(DirectExecutionMembership) * checkpoint.route_memberships.size(),
                     cudaMemcpyHostToDevice),
          "restore route memberships");
    }
    if (!checkpoint.packed_panels.empty()) {
      fab.packed_panel_count = static_cast<std::uint32_t>(checkpoint.packed_panels.size());
      require_cuda(cudaMemcpy(fab.view.packed_panels, checkpoint.packed_panels.data(),
                              sizeof(DirectPackedSparsePanel) * checkpoint.packed_panels.size(),
                              cudaMemcpyHostToDevice),
                   "restore packed panels");
    }
    if (!checkpoint.packed_source_meta.empty()) {
      require_cuda(cudaMemcpy(fab.view.packed_source_meta, checkpoint.packed_source_meta.data(),
                              sizeof(DirectPackedSourceMeta) * checkpoint.packed_source_meta.size(),
                              cudaMemcpyHostToDevice),
                   "restore packed source meta");
    }
    if (!checkpoint.packed_entries.empty()) {
      fab.packed_entry_count = static_cast<std::uint32_t>(checkpoint.packed_entries.size());
      require_cuda(cudaMemcpy(fab.view.packed_entries, checkpoint.packed_entries.data(),
                              sizeof(DirectPackedEntry) * checkpoint.packed_entries.size(),
                              cudaMemcpyHostToDevice),
                   "restore packed entries");
    }
    if (!checkpoint.dense_tiles.empty()) {
      fab.dense_tile_count = static_cast<std::uint32_t>(checkpoint.dense_tiles.size());
      require_cuda(cudaMemcpy(fab.view.dense_tiles, checkpoint.dense_tiles.data(),
                              sizeof(DirectDenseTile) * checkpoint.dense_tiles.size(),
                              cudaMemcpyHostToDevice),
                   "restore dense tiles");
    }
    if (!checkpoint.tract_lanes.empty()) {
      fab.tract_lane_count = static_cast<std::uint32_t>(checkpoint.tract_lanes.size());
      require_cuda(cudaMemcpy(fab.view.tract_lanes, checkpoint.tract_lanes.data(),
                              sizeof(DirectTractLane) * checkpoint.tract_lanes.size(),
                              cudaMemcpyHostToDevice),
                   "restore tract lanes");
    }
    if (!checkpoint.tract_bucket_counts.empty()) {
      require_cuda(cudaMemcpy(fab.view.tract_bucket_counts, checkpoint.tract_bucket_counts.data(),
                              sizeof(std::uint32_t) * checkpoint.tract_bucket_counts.size(),
                              cudaMemcpyHostToDevice),
                   "restore tract bucket counts");
      for (std::uint32_t c : checkpoint.tract_bucket_counts)
        pending_tract_total += c;
    }
    if (!checkpoint.tract_ring_packets.empty()) {
      require_cuda(cudaMemcpy(fab.view.tract_ring_packets, checkpoint.tract_ring_packets.data(),
                              sizeof(DirectTractPacket) * checkpoint.tract_ring_packets.size(),
                              cudaMemcpyHostToDevice),
                   "restore tract ring packets");
      // #1208: the second and last way a packet enters the ring. Without this a
      // restored adult would carry in-flight transport the step is entitled to
      // ignore -- the checkpoint would silently lose it.
      fab.tract_ring_ever_written = true;
    }
  }

  runtime.frontier_launch_bound =
      std::min(frontier_count + pending_tract_total, runtime.frontier_capacity);
  runtime.eligibility_launch_bound = std::min(eligibility_count, runtime.eligibility_capacity);

  // #1179 representation state.
  if (runtime.representation != nullptr) {
    DirectRepresentationRuntime& rep = *runtime.representation;
    if (!checkpoint.representation_source_state.empty()) {
      require_cuda(cudaMemcpy(rep.view.source_state, checkpoint.representation_source_state.data(),
                              sizeof(DirectSourceRepresentationState) * rep.node_count,
                              cudaMemcpyHostToDevice),
                   "restore representation source state");
    }
    if (!checkpoint.representation_resident_entries.empty()) {
      require_cuda(
          cudaMemcpy(rep.resident_entries, checkpoint.representation_resident_entries.data(),
                    sizeof(DirectPackedEntry) * checkpoint.representation_resident_entries.size(),
                    cudaMemcpyHostToDevice),
          "restore representation resident entries");
    }
    if (!checkpoint.representation_resident_entry_count.empty()) {
      require_cuda(cudaMemcpy(rep.resident_entry_count,
                              checkpoint.representation_resident_entry_count.data(),
                              sizeof(std::uint32_t) * rep.node_count, cudaMemcpyHostToDevice),
                   "restore representation resident entry counts");
    }
    if (!checkpoint.representation_state_owners.empty()) {
      require_cuda(cudaMemcpy(rep.view.state_owners, checkpoint.representation_state_owners.data(),
                              sizeof(DirectRepresentationStateOwner) * rep.state_owner_capacity,
                              cudaMemcpyHostToDevice),
                   "restore representation state owners");
    }
  }

  const Root256 restored_brain_root = direct_brain_state_root(runtime.brain);
  const Root256 restored_representation_root = runtime.representation != nullptr
      ? direct_representation_state_root(*runtime.representation)
      : Root256{};
  struct CombinedRoot { Root256 brain; Root256 representation; }
      restored_combined{restored_brain_root, restored_representation_root};
  if (direct_network::recipe::content_root(&restored_combined, sizeof(restored_combined)) !=
      checkpoint.learned_state_root) {
    destroy_direct_adult_runtime(&runtime, true);
    throw std::runtime_error("restored direct adult state root differs");
  }

  // #1184 checkpoint-pending-I/O: rehydrate the bridge's outstanding ticket
  // ancestry so a return that arrives after restart can still find the
  // (cue_node, context) it needs to address the membrane at, and so a
  // stale/replayed return for an already-settled ticket still finds
  // nothing to bind to (no double credit).
  if (bridge_out != nullptr) {
    DirectCausalActionBridge::BridgeCheckpointState pending;
    pending.outstanding = checkpoint.bridge_outstanding_tickets;
    pending.completed = checkpoint.bridge_completed_returns;
    try {
      bridge_out->restore_pending_state(pending);
    } catch (...) {
      destroy_direct_adult_runtime(&runtime, true);
      throw;
    }
  }
  return runtime;
}

}  // namespace substrate::direct_adult
