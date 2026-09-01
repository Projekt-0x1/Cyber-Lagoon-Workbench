#include "hardware_native/direct_adult_persistent_phases.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_dense_execution.cuh"
#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_bounded_sm_workspace.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"

namespace substrate::direct_adult_core {

__device__ __noinline__ void persistent_development_phase_device(
    cooperative_groups::grid_group grid,
    const ResidentAdultEpochSnapshot& epoch_snap,
    std::uint32_t current_tick,
    std::uint32_t global_tid,
    std::uint32_t total_threads,
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint32_t* route_causal_pin_bits,
    AdultExecutionConfig config,
    ResidentDevelopmentWorkspace resident_development,
    DirectBrain brain) {
    // 4. Γ-authored resident development is part of the same adult clock in
    // persistent mode. The proposal, target arbitration, admission prefix,
    // and commit functions are the same device functions used by the
    // host-stepped wrapper; this phase has no host callback or readback.
    const std::uint32_t* development_pins = rebuild_persistent_route_causal_pins(
        grid, eligibility_table, live_eligibility_count, brain.route_incarnations,
        route_capacity, epoch_snap.tick, route_causal_pin_bits, global_tid,
        total_threads, config.honor_causal_pin, nullptr);
    if (resident_development.counters != nullptr &&
        resident_development.block_sums != nullptr &&
        resident_development.block_offsets != nullptr &&
        resident_development.history_prefix != nullptr &&
        resident_development.topology_plan != nullptr &&
        brain.development != nullptr) {
      if (grid.thread_rank() == 0u) {
        direct_network::resident_development_advance_epoch(
            brain.development, resident_development.reserve_snapshot);
        *resident_development.counters = direct_network::ResidentDevelopmentCounters{};
        *resident_development.topology_plan = direct_network::TopologyCommitPlan{};
      }
      for (std::uint32_t i = global_tid; i < node_count; i += total_threads)
        resident_development.target_claims[i] = 0xffffffffffffffffULL;
      grid.sync();

      if (global_tid < node_count)
        direct_network::resident_development_mature_node(
            brain, global_tid, resident_development.counters);
      grid.sync();

      if (global_tid < node_count) {
        direct_network::resident_development_propose_node(
            brain, global_tid, resident_development.proposals,
            resident_development.costs, resident_development.target_claims,
            resident_development.counters);
      }
      grid.sync();

      if (global_tid < node_count) {
        direct_network::resident_development_filter_node(
            resident_development.proposals, resident_development.costs,
            resident_development.target_claims, global_tid, node_count);
      }
      grid.sync();

      // Exact node-order exclusive prefix, with block totals reduced by rank
      // order. This is the persistent-kernel equivalent of CUB's host-stepped
      // scan and is deterministic under block scheduling.
      __shared__ std::uint32_t local_scan[kDirectPersistentScanWorkspaceWords];
      persistent_exclusive_scan(
          grid, resident_development.costs,
          resident_development.cost_prefix, resident_development.block_sums,
          resident_development.block_offsets, local_scan, node_count, global_tid);

      if (global_tid < node_count) {
        direct_network::resident_development_prepare_topology_node(
            brain, global_tid, resident_development.proposals,
            resident_development.costs,
            resident_development.cost_prefix, resident_development.reserve_snapshot,
            resident_development.target_claims, development_pins,
            resident_development.counters, resident_development.topology_plan);
      }
      grid.sync();
      persistent_exclusive_scan(
          grid, resident_development.costs,
          resident_development.history_prefix, resident_development.block_sums,
          resident_development.block_offsets, local_scan, node_count, global_tid);
      if (global_tid < node_count)
        direct_network::resident_development_finalize_topology_node(
            brain, global_tid, resident_development.proposals,
            resident_development.costs, resident_development.history_prefix,
            resident_development.counters, resident_development.topology_plan);
      grid.sync();
      persistent_exclusive_scan(
          grid, resident_development.costs,
          resident_development.cost_prefix, resident_development.block_sums,
          resident_development.block_offsets, local_scan, node_count, global_tid);
      if (global_tid < node_count)
        direct_network::resident_development_admit_topology_work_node(
            brain, global_tid, resident_development.proposals,
            resident_development.costs, resident_development.cost_prefix,
            resident_development.counters, resident_development.topology_plan);
      grid.sync();
      persistent_exclusive_scan(
          grid, resident_development.costs,
          resident_development.history_prefix, resident_development.block_sums,
          resident_development.block_offsets, local_scan, node_count, global_tid);
      if (global_tid == 0u)
        direct_network::resident_development_begin_topology_transaction(
            brain, resident_development.topology_plan);
      grid.sync();
      if (global_tid < node_count)
        direct_network::resident_development_commit_topology_node(
            brain, global_tid, resident_development.proposals,
            resident_development.cost_prefix,
            resident_development.history_prefix,
            resident_development.reserve_snapshot,
            resident_development.target_claims, development_pins,
            resident_development.counters, resident_development.topology_plan);
      grid.sync();
      if (global_tid == 0u) {
        direct_network::resident_development_nominate_committed_retractions(
            brain, resident_development.proposals, resident_development.counters);
        direct_network::resident_development_finish_topology_transaction(
            brain, resident_development.topology_plan);
        direct_network::resident_development_commit_recipe_transaction(
            brain, resident_development.proposals, resident_development.counters);
        direct_network::resident_development_commit_construction_fronts(
            brain, resident_development.proposals, resident_development.costs,
            resident_development.reserve_snapshot, resident_development.counters);
        direct_network::resident_development_condense_postbirth_recipe(
            brain, resident_development.counters,
            resident_development.cause_memo);
        direct_network::resident_development_finish_admission_epoch(
            brain, resident_development.counters);
      }
      grid.sync();
    }

}

}  // namespace substrate::direct_adult_core
