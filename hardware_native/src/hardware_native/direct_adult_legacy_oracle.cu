#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_network_recipe_abi.cuh"
#include "hardware_native/direct_canonical_evaluator_device.cuh"
#include "hardware_native/direct_dynamic_topology_arena.cuh"
#include "hardware_native/direct_exact_eligibility_device.cuh"
#include "hardware_native/direct_execution_fabric.cuh"
#include "hardware_native/direct_implicit_causal_mesh.cuh"
#include "hardware_native/direct_adult_legacy_oracle.cuh"
#include "hardware_native/direct_resource_ecology_legacy.cuh"
#include "hardware_native/direct_retention_policy.cuh"
#include "hardware_native/direct_representation_compiler.cuh"

namespace substrate::direct_adult {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// The adult runtime is extracted whole into the stage units above; this
// translation unit keeps composition -- birth, ingress, settlement,
// propagation, state roots -- plus receipt packing and every launch site.
#include "direct_adult_legacy_birth.cuh"
#include "direct_adult_legacy_membrane_ingress.cuh"
#include "direct_adult_eligibility_settlement.cuh"
#include "direct_adult_frontier_propagation.cuh"
#include "direct_adult_state_root.cuh"
__global__ void pack_adult_step_receipt_kernel(const AdultCounters* counters,
                                               const std::uint32_t* frontier_count,
                                               const std::uint32_t* motor_count,
                                               const std::uint32_t* eligibility_count,
                                               const std::uint32_t* tract_bucket_counts,
                                               std::uint32_t max_tract_delay,
                                               AdultStepReceipt* receipt) {
  if (blockIdx.x != 0)
    return;
  constexpr std::uint32_t kCounterWords = sizeof(AdultCounters) / sizeof(std::uint32_t);
  const std::uint32_t lane = threadIdx.x;
  if (lane < kCounterWords) {
    reinterpret_cast<std::uint32_t*>(&receipt->counters)[lane] =
        reinterpret_cast<const std::uint32_t*>(counters)[lane];
  } else if (lane == kCounterWords) {
    receipt->next_frontier_size = *frontier_count;
  } else if (lane == kCounterWords + 1u) {
    receipt->motor_count = *motor_count;
  } else if (lane == kCounterWords + 2u) {
    receipt->eligibility_count = *eligibility_count;
  } else if (lane == kCounterWords + 3u) {
    std::uint32_t pending = 0u;
    if (tract_bucket_counts != nullptr) {
      for (std::uint32_t b = 0; b <= max_tract_delay; ++b)
        pending += tract_bucket_counts[b];
    }
    receipt->pending_tract_packets = pending;
  }
}
std::uint32_t grid_for(std::uint32_t count, std::uint32_t block = 128u) {
  return std::max(1u, (count + block - 1u) / block);
}

std::uint32_t next_power_of_two(std::uint32_t value) {
  value = std::max(2u, value) - 1u;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1u;
}
}  // namespace

Root256 direct_brain_state_root(const DirectBrainV01& brain) {
  return root_over_brain_arrays(brain);
}

BirthReceiptV0 compile_direct_brain_v01(const Gamma& gamma, const BodyManifestV0& body,
                                        DirectBrainV01* out_brain) {
  if (out_brain == nullptr)
    throw std::invalid_argument("out_brain cannot be null");
  const auto begin = std::chrono::steady_clock::now();
  DirectBrainV01 brain{};
  brain.genome_root = canonical_genome_root(gamma);
  brain.body_root = content_root(&body, sizeof(body));
  brain.territory_count = gamma.header.seed_count;

  Gamma* device_gamma = nullptr;
  check_cuda(cudaMalloc(&device_gamma, sizeof(Gamma)), "allocate device gamma");
  check_cuda(cudaMemcpy(device_gamma, &gamma, sizeof(Gamma), cudaMemcpyHostToDevice),
             "copy device gamma");

  const std::uint32_t plan_count = std::max(1u, gamma.header.seed_count);
  TerritoryPlan* device_plans = nullptr;
  check_cuda(cudaMalloc(&device_plans, sizeof(TerritoryPlan) * plan_count),
             "allocate territory plans");
  plan_territories_kernel<<<grid_for(plan_count), 128>>>(device_gamma, device_plans, plan_count);
  check_cuda(cudaGetLastError(), "plan territories");

  std::vector<TerritoryPlan> host_plans(plan_count);
  check_cuda(cudaMemcpy(host_plans.data(), device_plans, sizeof(TerritoryPlan) * plan_count,
                        cudaMemcpyDeviceToHost),
             "read territory plans");

  std::vector<std::uint32_t> host_node_counts(plan_count, 0u);
  std::vector<std::uint32_t> host_route_counts(plan_count, 0u);
  std::uint32_t node_count = 0u;
  std::uint32_t route_count = 0u;
  std::uint64_t logical_routes = 0u;
  std::uint64_t virtual_routes = 0u;
  std::vector<DirectImplicitFamily> implicit_families(plan_count);

  for (std::size_t i = 0; i < host_plans.size(); ++i) {
    if (host_plans[i].active == 0u)
      continue;
    host_node_counts[i] = host_plans[i].node_count;
    host_route_counts[i] = host_plans[i].route_count;
    node_count += host_plans[i].node_count;
    route_count += host_plans[i].route_count;
    logical_routes += host_plans[i].logical_route_count;
    virtual_routes += host_plans[i].virtual_route_count;

    DirectImplicitFamily family{};
    family.node_begin = host_plans[i].node_offset;
    family.node_count = host_plans[i].node_count;
    family.local_degree = host_plans[i].local_degree;
    family.chord_stride = host_plans[i].chord_stride;
    family.lineage = host_plans[i].lineage;
    family.first_virtual_slot = std::min<std::uint32_t>(2u, host_plans[i].local_degree);
    family.virtual_slot_count = host_plans[i].local_degree > family.first_virtual_slot
                                    ? (host_plans[i].local_degree - family.first_virtual_slot)
                                    : 0u;
    family.flags = family.virtual_slot_count != 0u ? kImplicitFamilySkipLastSlotForFirstNode : 0u;
    family.base_conductance_q16 = kConductanceOneQ16;
    family.coeff_q12[0] = 0;
    family.coeff_q12[1] = 0;
    family.coeff_q12[2] = 0;
    family.coeff_q12[3] = 0;
    implicit_families[i] = family;
  }

  std::vector<std::uint32_t> host_node_offsets(plan_count, 0u);
  std::vector<std::uint32_t> host_route_offsets(plan_count, 0u);
  std::uint32_t node_running = 0u;
  std::uint32_t route_running = 0u;
  for (std::size_t i = 0; i < host_plans.size(); ++i) {
    host_node_offsets[i] = node_running;
    host_route_offsets[i] = route_running;
    implicit_families[i].node_begin = node_running;
    node_running += host_node_counts[i];
    route_running += host_route_counts[i];
  }

  std::uint32_t* device_node_offsets = nullptr;
  std::uint32_t* device_route_offsets = nullptr;
  check_cuda(cudaMalloc(&device_node_offsets, sizeof(std::uint32_t) * plan_count),
             "allocate node offsets");
  check_cuda(cudaMalloc(&device_route_offsets, sizeof(std::uint32_t) * plan_count),
             "allocate route offsets");
  check_cuda(cudaMemcpy(device_node_offsets, host_node_offsets.data(),
                        sizeof(std::uint32_t) * plan_count, cudaMemcpyHostToDevice),
             "copy node offsets");
  check_cuda(cudaMemcpy(device_route_offsets, host_route_offsets.data(),
                        sizeof(std::uint32_t) * plan_count, cudaMemcpyHostToDevice),
             "copy route offsets");
  install_offsets_kernel<<<grid_for(plan_count), 128>>>(device_plans, device_node_offsets,
                                                        device_route_offsets, plan_count);
  check_cuda(cudaGetLastError(), "install offsets");

  const std::uint32_t affordable_route_capacity =
      gamma.header.matter_budget > node_count ? (gamma.header.matter_budget - node_count) : 0u;
  const std::uint32_t desired_growth = std::max(1024u, node_count * 2u);
  const std::uint32_t route_capacity = std::max(
      route_count + 32u, std::min(affordable_route_capacity, route_count + desired_growth));
  const std::uint32_t context_index_capacity =
      next_power_of_two(std::max(1024u, route_capacity * 2u));
  const std::uint64_t reserve =
      affordable_route_capacity > route_count ? (affordable_route_capacity - route_count) : 0ull;

  brain.node_count = node_count;
  brain.route_count = route_count;
  brain.route_capacity = route_capacity;
  brain.logical_route_count = logical_routes;
  brain.virtual_route_count = virtual_routes;
  brain.recurrent_route_count = static_cast<std::uint32_t>(logical_routes);
  brain.long_tract_count = gamma.header.seed_count;
  brain.context_index_capacity = next_power_of_two(std::max(1024u, route_capacity * 2u));

  check_cuda(cudaMalloc(&brain.nodes, sizeof(DirectNode) * node_count), "allocate direct nodes");
  check_cuda(cudaMalloc(&brain.routes, sizeof(DirectRoute) * route_capacity),
             "allocate direct routes");
  check_cuda(cudaMemset(brain.routes, 0, sizeof(DirectRoute) * route_capacity),
             "clear direct routes");
  check_cuda(cudaMalloc(&brain.development, sizeof(ResidentDevelopmentState)),
             "allocate resident development");
  check_cuda(cudaMalloc(&brain.live_route_count, sizeof(std::uint32_t)),
             "allocate live route count");
  check_cuda(cudaMalloc(&brain.context_index,
                         sizeof(ContextRouteIndexEntry) * brain.context_index_capacity),
             "allocate context route index");
  check_cuda(cudaMemset(brain.context_index, 0xff,
                         sizeof(ContextRouteIndexEntry) * brain.context_index_capacity),
             "clear context route index");
  check_cuda(cudaMalloc(&brain.resource_ecology, sizeof(DirectResourceEcologyState)),
             "allocate direct resource ecology");
  check_cuda(cudaMalloc(&brain.retention_bank, sizeof(DirectRetentionState) * route_capacity),
             "allocate direct retention bank");
  check_cuda(cudaMalloc(&brain.minimal_retention_bank, sizeof(DirectMinimalRetentionState) * route_capacity),
             "allocate direct minimal retention bank");

  ResidentDevelopmentState initial_development{
      0u, gamma.header.development_end_tick, 1u << 16, 1u << 14, reserve, 0u, 0ull};
  check_cuda(cudaMemcpy(brain.development, &initial_development, sizeof(initial_development),
                        cudaMemcpyHostToDevice),
             "initialize resident development");
  check_cuda(
      cudaMemcpy(brain.live_route_count, &route_count, sizeof(route_count), cudaMemcpyHostToDevice),
      "initialize live route count");

  DirectResourceEcologyState initial_ecology{};
  initialize_direct_resource_ecology_state(&initial_ecology);
  {
    auto pool = [&initial_ecology](DirectResourcePoolKind kind) -> DirectResourcePoolState& {
      return initial_ecology.pools[static_cast<std::uint32_t>(kind)];
    };
    // capacity_units bounds a pool against starvation by its neighbours;
    // bytes_per_unit converts that bound into the physical currency the global
    // budget is denominated in. A pool with bytes_per_unit == 0 is declared to
    // consume no arena and is excluded from the global bound -- which is only
    // honest for pools that own no device allocation, so keep it explicit.
    DirectResourcePoolState& nodes = pool(DirectResourcePoolKind::node_state);
    nodes.capacity_units = node_count;
    nodes.live_units = node_count;
    nodes.charged_units = node_count;
    nodes.high_water_units = node_count;
    nodes.bytes_per_unit = sizeof(DirectNode);

    DirectResourcePoolState& routes = pool(DirectResourcePoolKind::explicit_interaction);
    routes.capacity_units = route_capacity;
    routes.bytes_per_unit = sizeof(DirectRoute) + sizeof(DirectRouteSlotMeta) +
                            sizeof(DirectRetentionState) + sizeof(DirectMinimalRetentionState);
    // live/charged are seeded from physical matter below, not asserted here.

    DirectResourcePoolState& implicit_pool = pool(DirectResourcePoolKind::implicit_exception);
    implicit_pool.capacity_units = std::max(1024u, node_count / 4u);
    implicit_pool.bytes_per_unit = sizeof(DirectImplicitException);

    DirectResourcePoolState& contexts = pool(DirectResourcePoolKind::context_record);
    contexts.capacity_units = brain.context_index_capacity;
    contexts.bytes_per_unit = sizeof(ContextRouteIndexEntry);

    DirectResourcePoolState& proposals = pool(DirectResourcePoolKind::topology_proposal);
    proposals.capacity_units = 8192u;

    // The global bound is the sum of what the declared pools can physically
    // occupy at full capacity. Sized this way it is not yet a binding constraint
    // at birth -- it becomes one the moment a pool is given a capacity the arena
    // cannot cover, which is exactly the pressure case the authority contract
    // drives. Its value is that it EXISTS as one authoritative number: fourteen
    // independent per-pool bounds can each be green while their sum overcommits
    // the device.
    std::uint64_t declared_bytes = 0u;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(DirectResourcePoolKind::count); ++i)
      declared_bytes += initial_ecology.pools[i].capacity_units *
                        initial_ecology.pools[i].bytes_per_unit;
    initial_ecology.global_capacity_bytes = declared_bytes;
  }
  check_cuda(cudaMemcpy(brain.resource_ecology, &initial_ecology, sizeof(initial_ecology),
                        cudaMemcpyHostToDevice),
             "initialize direct resource ecology");
  initialize_direct_retention_bank(brain.retention_bank, route_capacity);
  initialize_direct_minimal_retention_bank(brain.minimal_retention_bank, route_capacity);

  materialize_territories_kernel<<<plan_count, 128>>>(device_plans, plan_count, brain.nodes,
                                                      brain.routes);
  check_cuda(cudaGetLastError(), "materialize territories");

  attach_boundaries_kernel<<<grid_for(body.binding_count), 128>>>(brain.nodes, device_plans,
                                                                  plan_count, body);
  check_cuda(cudaGetLastError(), "attach boundaries");

  initialize_direct_topology_state(&brain);
  initialize_direct_implicit_state(&brain, implicit_families.data(), gamma.header.seed_count,
                                   virtual_routes, std::max(1024u, node_count / 4u));
  // Seed the ledger from matter that already exists, exactly once. From this
  // point on live_units is written only by the allocators charging through the
  // ledger; nothing re-derives it from the free list.
  seed_resource_pool_ledgers_from_matter(&brain);
  check_cuda(cudaDeviceSynchronize(), "seed resource pool ledgers");

  cudaFree(device_node_offsets);
  cudaFree(device_route_offsets);
  cudaFree(device_plans);
  cudaFree(device_gamma);

  brain.birth_root = direct_brain_state_root(brain);
  *out_brain = brain;

  const double gestation_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  BirthReceiptV0 receipt{};
  receipt.genome_root = brain.genome_root;
  receipt.body_root = brain.body_root;
  receipt.birth_root = brain.birth_root;
  receipt.node_count = node_count;
  receipt.route_count = static_cast<std::uint32_t>(logical_routes);
  receipt.territory_count = brain.territory_count;
  receipt.recurrent_route_count = brain.recurrent_route_count;
  receipt.long_tract_count = brain.long_tract_count;
  receipt.device_bytes = sizeof(DirectNode) * node_count + sizeof(DirectRoute) * route_capacity +
                         sizeof(ContextRouteIndexEntry) * brain.context_index_capacity +
                         sizeof(DirectRouteSlotMeta) * route_capacity +
                         sizeof(std::uint32_t) * (route_capacity + node_count) +
                         sizeof(DirectImplicitFamily) * brain.implicit.family_count +
                         sizeof(DirectImplicitException) * brain.implicit.exception_capacity +
                         sizeof(DirectResourceEcologyState) +
                         sizeof(DirectRetentionState) * route_capacity +
                         sizeof(DirectMinimalRetentionState) * route_capacity;
  receipt.gestation_ms = static_cast<float>(gestation_seconds * 1000.0);
  receipt.compact_recipe = true;
  receipt.final_connectome_loaded = false;
  receipt.life_function_detached = true;
  receipt.explicit_route_count = route_count;
  receipt.implicit_family_count = brain.implicit.family_count;
  receipt.virtual_route_count = virtual_routes;
  return receipt;
}

void destroy_direct_brain(DirectBrainV01* brain) {
  if (brain == nullptr)
    return;
  destroy_direct_implicit_state(brain);
  destroy_direct_topology_state(brain);
  cudaFree(brain->minimal_retention_bank);
  cudaFree(brain->retention_bank);
  cudaFree(brain->resource_ecology);
  cudaFree(brain->nodes);
  cudaFree(brain->routes);
  cudaFree(brain->development);
  cudaFree(brain->live_route_count);
  cudaFree(brain->context_index);
  *brain = DirectBrainV01{};
}

DirectAdultRuntime create_direct_adult_runtime(DirectBrainV01 brain,
                                               std::uint32_t frontier_capacity,
                                               std::uint32_t motor_capacity,
                                               std::uint32_t eligibility_capacity_override) {
  DirectAdultRuntime runtime{};
  runtime.brain = brain;
  runtime.frontier_capacity = frontier_capacity;
  runtime.motor_capacity = motor_capacity;
  const std::uint32_t automatic_eligibility_capacity =
      std::min<std::uint32_t>(std::max(frontier_capacity * kEligibilityLaunchExpansion, 2048u),
                              std::max(brain.route_capacity, 2048u));
  runtime.eligibility_capacity = eligibility_capacity_override == 0u
                                     ? automatic_eligibility_capacity
                                     : std::max(1u, eligibility_capacity_override);
  runtime.eligibility_bucket_count = 1u;
  while (runtime.eligibility_bucket_count < runtime.eligibility_capacity * 2u)
    runtime.eligibility_bucket_count <<= 1u;
  runtime.frontier_launch_bound = 0u;
  runtime.return_credit_launch_block = 128u;
  runtime.eligibility_launch_bound = 0u;

  check_cuda(cudaMalloc(&runtime.frontier, sizeof(ActivityEvent) * frontier_capacity),
             "allocate frontier");
  check_cuda(cudaMalloc(&runtime.next_frontier, sizeof(ActivityEvent) * frontier_capacity),
             "allocate next frontier");
  check_cuda(
      cudaMalloc(&runtime.frontier_authority, sizeof(DirectIngressAuthority) * frontier_capacity),
      "allocate frontier authority");
  check_cuda(cudaMalloc(&runtime.next_frontier_authority,
                        sizeof(DirectIngressAuthority) * frontier_capacity),
             "allocate next frontier authority");
  check_cuda(cudaMalloc(&runtime.frontier_count, sizeof(std::uint32_t)), "allocate frontier count");
  check_cuda(cudaMalloc(&runtime.next_frontier_count, sizeof(std::uint32_t)),
             "allocate next frontier count");
  check_cuda(cudaMalloc(&runtime.motor_events, sizeof(MotorEvent) * motor_capacity),
             "allocate motor events");
  check_cuda(cudaMalloc(&runtime.motor_count, sizeof(std::uint32_t)), "allocate motor count");
  check_cuda(cudaMalloc(&runtime.ingress_staging, sizeof(ActivityEvent) * frontier_capacity),
             "allocate ingress staging");
  check_cuda(cudaMalloc(&runtime.bridge_mark_staging, sizeof(BridgeTicketMark) * motor_capacity),
             "allocate bridge ticket mark staging");
  check_cuda(cudaMalloc(&runtime.eligibility_bank,
                        sizeof(DirectEligibilityRecord) * runtime.eligibility_capacity),
             "allocate eligibility bank");
  check_cuda(cudaMalloc(&runtime.next_eligibility_bank,
                        sizeof(DirectEligibilityRecord) * runtime.eligibility_capacity),
             "allocate next eligibility bank");
  check_cuda(cudaMalloc(&runtime.eligibility_count, sizeof(std::uint32_t)),
             "allocate eligibility count");
  check_cuda(cudaMalloc(&runtime.next_eligibility_count, sizeof(std::uint32_t)),
             "allocate next eligibility count");
  check_cuda(cudaMalloc(&runtime.eligibility_bucket_heads,
                        sizeof(std::uint32_t) * runtime.eligibility_bucket_count),
             "allocate eligibility bucket heads");
  check_cuda(cudaMalloc(&runtime.next_eligibility_bucket_heads,
                        sizeof(std::uint32_t) * runtime.eligibility_bucket_count),
             "allocate next eligibility bucket heads");
  check_cuda(cudaMalloc(&runtime.eligibility_batch_admit, sizeof(std::uint32_t)),
             "allocate eligibility admission flag");
  check_cuda(cudaMalloc(&runtime.counters, sizeof(AdultCounters)), "allocate counters");
  check_cuda(cudaMalloc(&runtime.context_state, sizeof(ResidentContextState)),
             "allocate resident context");
  check_cuda(cudaMalloc(&runtime.step_receipt, sizeof(AdultStepReceipt)),
             "allocate adult step receipt");

  check_cuda(cudaMemset(runtime.frontier_count, 0, sizeof(std::uint32_t)), "clear frontier count");
  check_cuda(cudaMemset(runtime.next_frontier_count, 0, sizeof(std::uint32_t)),
             "clear next frontier count");
  check_cuda(
      cudaMemset(runtime.frontier_authority, 0, sizeof(DirectIngressAuthority) * frontier_capacity),
      "clear frontier authority");
  check_cuda(cudaMemset(runtime.next_frontier_authority, 0,
                        sizeof(DirectIngressAuthority) * frontier_capacity),
             "clear next frontier authority");
  check_cuda(cudaMemset(runtime.motor_count, 0, sizeof(std::uint32_t)), "clear motor count");
  check_cuda(cudaMemset(runtime.eligibility_count, 0, sizeof(std::uint32_t)),
             "clear eligibility count");
  check_cuda(cudaMemset(runtime.next_eligibility_count, 0, sizeof(std::uint32_t)),
             "clear next eligibility count");
  check_cuda(cudaMemset(runtime.counters, 0, sizeof(AdultCounters)), "clear counters");
  check_cuda(cudaMemset(runtime.eligibility_bank, 0,
                        sizeof(DirectEligibilityRecord) * runtime.eligibility_capacity),
             "clear eligibility bank");
  check_cuda(cudaMemset(runtime.next_eligibility_bank, 0,
                        sizeof(DirectEligibilityRecord) * runtime.eligibility_capacity),
             "clear next eligibility bank");
  check_cuda(cudaMemset(runtime.eligibility_bucket_heads, 0xff,
                        sizeof(std::uint32_t) * runtime.eligibility_bucket_count),
             "clear eligibility bucket heads");
  check_cuda(cudaMemset(runtime.next_eligibility_bucket_heads, 0xff,
                        sizeof(std::uint32_t) * runtime.eligibility_bucket_count),
             "clear next eligibility bucket heads");
  const std::uint32_t admit_initial = 1u;
  check_cuda(cudaMemcpy(runtime.eligibility_batch_admit, &admit_initial, sizeof(admit_initial),
                        cudaMemcpyHostToDevice),
             "initialize eligibility admission");

  const ResidentContextState initial_context{kInvalidIndex, 0u, kInvalidIndex, 0u, 0u, 0u, 0u, 0u};
  check_cuda(cudaMemcpy(runtime.context_state, &initial_context, sizeof(initial_context),
                        cudaMemcpyHostToDevice),
             "initialize resident context");

  ResidentDevelopmentState development{};
  check_cuda(
      cudaMemcpy(&development, brain.development, sizeof(development), cudaMemcpyDeviceToHost),
      "read resident developmental clock");
  runtime.tick = development.developmental_tick;
  runtime.topology_runtime = create_direct_topology_runtime(
      brain, runtime.frontier_capacity * kMaxImplicitActiveFanout + runtime.eligibility_capacity);
  runtime.fabric = create_direct_execution_fabric(brain.node_count, brain.route_capacity,
                                                  runtime.frontier_capacity);
  runtime.resource_maintenance = create_resource_maintenance_runtime(256u);

  // Computed here (not inside create_direct_representation_runtime) so the
  // exact allocated size can be charged to the ecology ledger below using the
  // SAME numbers that actually get cudaMalloc'd, rather than recomputing the
  // hint-to-power-of-2 rounding a second time and risking the two drifting
  // apart. Passing an already-rounded capacity into create_direct_representation_
  // runtime's own rounding loop is a no-op.
  const std::uint32_t representation_node_count = std::max<std::uint32_t>(brain.node_count, 1u);
  const std::uint32_t representation_state_owner_hint =
      std::max<std::uint32_t>(brain.route_capacity / 4u, 64u);
  std::uint32_t representation_state_owner_capacity = 64u;
  while (representation_state_owner_capacity < representation_state_owner_hint)
    representation_state_owner_capacity <<= 1u;

  // Some pools are sized when the runtime is created, not at birth, so their
  // bounds cannot be declared in compile_direct_brain_v01. Declare them here,
  // once, before any step runs -- a pool whose capacity is still zero when the
  // first allocation arrives would refuse everything.
  if (brain.resource_ecology != nullptr) {
    DirectResourceEcologyState ecology{};
    check_cuda(cudaMemcpy(&ecology, brain.resource_ecology, sizeof(ecology),
                          cudaMemcpyDeviceToHost),
               "read resource ecology for runtime pool sizing");
    // Declare, never re-derive.
    //
    // A restored runtime arrives with these pools already carrying checkpointed
    // capacities, and recomputing them here silently overwrote restored state --
    // which showed up as a restored organism diverging from the original when both
    // were run forward, in `global_capacity_bytes`. Only an undeclared pool
    // (capacity still zero, i.e. a freshly compiled brain) is sized here.
    DirectResourcePoolState& eligibility =
        ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::eligibility_record)];
    if (eligibility.capacity_units == 0u)
      eligibility.capacity_units = runtime.eligibility_capacity;
    // Double-buffered: compaction re-appends survivors into a second bank, so
    // both banks are physically resident and the peak overlap is real matter.
    eligibility.bytes_per_unit = 2u * sizeof(DirectEligibilityRecord);
    DirectResourcePoolState& proposals =
        ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::topology_proposal)];
    if (proposals.capacity_units == 0u)
      proposals.capacity_units =
          runtime.frontier_capacity * kMaxImplicitActiveFanout + runtime.eligibility_capacity;
    proposals.bytes_per_unit = sizeof(DirectTopologyProposal);
    // #1179 P0.1: create_direct_representation_runtime (called just below)
    // does real cudaMalloc for these arrays, once, in full, at creation --
    // never grown incrementally. Before this the representation compiler
    // owned real resident silicon invisible to global_capacity_bytes: bounded
    // by cudaMalloc, not bounded with the rest of the organism. Charged in
    // full immediately, matching the `nodes` pool's own precedent at
    // gestation (capacity == live == charged from the moment the array
    // exists), not a reserve/commit dance -- there is no partial-occupancy
    // state for an array that is either fully allocated or does not exist.
    DirectResourcePoolState& rep_arrays = ecology.pools[static_cast<std::uint32_t>(
        DirectResourcePoolKind::representation_source_state)];
    const bool rep_arrays_freshly_declared = rep_arrays.capacity_units == 0u;
    if (rep_arrays_freshly_declared) {
      rep_arrays.capacity_units = representation_node_count;
      rep_arrays.live_units = representation_node_count;
      rep_arrays.charged_units = representation_node_count;
      rep_arrays.high_water_units = representation_node_count;
      rep_arrays.bytes_per_unit = sizeof(DirectSourceRepresentationState) +
                                  sizeof(std::uint32_t) +  // claim_ordinal
                                  kRepresentationResidentReserve * sizeof(DirectPackedEntry) +
                                  sizeof(std::uint32_t);   // resident_entry_count
    }
    DirectResourcePoolState& rep_owners = ecology.pools[static_cast<std::uint32_t>(
        DirectResourcePoolKind::representation_state_owner)];
    const bool rep_owners_freshly_declared = rep_owners.capacity_units == 0u;
    if (rep_owners_freshly_declared) {
      rep_owners.capacity_units = representation_state_owner_capacity;
      // #1179 P0.1 items 5/6: unlike rep_arrays (every node's slot always
      // exists), a state-owner slot is a shared pool -- charged and reserved
      // in full the moment the backing array exists (the bytes are real
      // either way), but live_units tracks actual occupancy, moved from
      // reserved to live by device_commit_pool_units on each free->shadow
      // claim in find_or_create_state_owner and back by device_uncommit_
      // pool_units on each return to free, so a receipt can read live
      // logical owners rather than array capacity.
      rep_owners.reserved_units = representation_state_owner_capacity;
      rep_owners.charged_units = representation_state_owner_capacity;
      rep_owners.high_water_units = representation_state_owner_capacity;
      rep_owners.bytes_per_unit = sizeof(DirectRepresentationStateOwner);
    }
    // global_charged_bytes has no other writer for these two pools (unlike
    // capacity_units below, nothing recomputes it from a sum), so add their
    // charge unconditionally, once, the moment they are first declared.
    if (rep_arrays_freshly_declared)
      ecology.global_charged_bytes +=
          static_cast<std::uint64_t>(representation_node_count) * rep_arrays.bytes_per_unit;
    if (rep_owners_freshly_declared)
      ecology.global_charged_bytes += static_cast<std::uint64_t>(representation_state_owner_capacity) *
                                      rep_owners.bytes_per_unit;
    // `global_capacity_bytes == 0` here means no fixture/gestation call has
    // ever set it -- the branch below recomputes it as the sum of every
    // pool's capacity, and rep_arrays/rep_owners' fields are already set
    // above, so that sum already counts them; adding again here would
    // double-charge. Only when the ceiling was already nonzero (the normal
    // gestated-brain case, set once at birth before these two pools existed)
    // does their contribution still need to be added on top.
    if (ecology.global_capacity_bytes == 0u) {
      std::uint64_t declared_bytes = 0u;
      for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(DirectResourcePoolKind::count); ++i)
        declared_bytes += ecology.pools[i].capacity_units * ecology.pools[i].bytes_per_unit;
      ecology.global_capacity_bytes = declared_bytes;
    } else {
      if (rep_arrays_freshly_declared)
        ecology.global_capacity_bytes +=
            static_cast<std::uint64_t>(representation_node_count) * rep_arrays.bytes_per_unit;
      if (rep_owners_freshly_declared)
        ecology.global_capacity_bytes +=
            static_cast<std::uint64_t>(representation_state_owner_capacity) *
            rep_owners.bytes_per_unit;
    }
    check_cuda(cudaMemcpy(brain.resource_ecology, &ecology, sizeof(ecology),
                          cudaMemcpyHostToDevice),
               "publish runtime-sized resource pools");
  }
  runtime.representation = create_direct_representation_runtime(
      representation_node_count, representation_state_owner_capacity);
  return runtime;
}

void destroy_direct_adult_runtime(DirectAdultRuntime* runtime, bool destroy_brain) {
  if (runtime == nullptr)
    return;
  destroy_resource_maintenance_runtime(runtime->resource_maintenance);
  destroy_direct_topology_runtime(runtime->topology_runtime);
  destroy_direct_execution_fabric(runtime->fabric);
  // #1179 P0.1 requirement 5: destruction restores the exact global charge.
  // Read the pool's own recorded capacity_units rather than re-deriving from
  // runtime->representation's fields, so release is exact even if a future
  // caller changes how the runtime sizes those arrays -- the ledger entry is
  // the source of truth for what was charged, not a second computation of it.
  // Representation is destroyed unconditionally below regardless of
  // destroy_brain, so its charge must be released unconditionally too;
  // brain.resource_ecology is still valid device memory here either way,
  // since destroy_direct_brain (which frees it) runs later and only if
  // destroy_brain is set.
  if (runtime->representation != nullptr && runtime->brain.resource_ecology != nullptr) {
    DirectResourceEcologyState ecology{};
    check_cuda(cudaMemcpy(&ecology, runtime->brain.resource_ecology, sizeof(ecology),
                          cudaMemcpyDeviceToHost),
               "read resource ecology for representation teardown");
    DirectResourcePoolState& rep_arrays = ecology.pools[static_cast<std::uint32_t>(
        DirectResourcePoolKind::representation_source_state)];
    if (rep_arrays.capacity_units != 0u) {
      const std::uint64_t bytes =
          static_cast<std::uint64_t>(rep_arrays.capacity_units) * rep_arrays.bytes_per_unit;
      ecology.global_capacity_bytes -= std::min(bytes, ecology.global_capacity_bytes);
      ecology.global_charged_bytes -= std::min(bytes, ecology.global_charged_bytes);
      rep_arrays = DirectResourcePoolState{};
    }
    DirectResourcePoolState& rep_owners = ecology.pools[static_cast<std::uint32_t>(
        DirectResourcePoolKind::representation_state_owner)];
    if (rep_owners.capacity_units != 0u) {
      const std::uint64_t bytes =
          static_cast<std::uint64_t>(rep_owners.capacity_units) * rep_owners.bytes_per_unit;
      ecology.global_capacity_bytes -= std::min(bytes, ecology.global_capacity_bytes);
      ecology.global_charged_bytes -= std::min(bytes, ecology.global_charged_bytes);
      rep_owners = DirectResourcePoolState{};
    }
    check_cuda(cudaMemcpy(runtime->brain.resource_ecology, &ecology, sizeof(ecology),
                          cudaMemcpyHostToDevice),
               "publish representation teardown resource release");
  }
  destroy_direct_representation_runtime(runtime->representation);
  cudaFree(runtime->frontier);
  cudaFree(runtime->next_frontier);
  cudaFree(runtime->frontier_authority);
  cudaFree(runtime->next_frontier_authority);
  cudaFree(runtime->frontier_count);
  cudaFree(runtime->next_frontier_count);
  cudaFree(runtime->motor_events);
  cudaFree(runtime->motor_count);
  cudaFree(runtime->bridge_mark_staging);
  cudaFree(runtime->ingress_staging);
  cudaFree(runtime->eligibility_bank);
  cudaFree(runtime->next_eligibility_bank);
  cudaFree(runtime->eligibility_count);
  cudaFree(runtime->next_eligibility_count);
  cudaFree(runtime->eligibility_bucket_heads);
  cudaFree(runtime->next_eligibility_bucket_heads);
  cudaFree(runtime->eligibility_batch_admit);
  cudaFree(runtime->counters);
  cudaFree(runtime->context_state);
  cudaFree(runtime->step_receipt);
  if (destroy_brain)
    destroy_direct_brain(&runtime->brain);
  *runtime = DirectAdultRuntime{};
}

void inject_raw_event(DirectAdultRuntime* runtime, const ActivityEvent& event) {
  if (runtime == nullptr)
    return;
  if (runtime->frontier_launch_bound < runtime->frontier_capacity)
    ++runtime->frontier_launch_bound;
  append_event_kernel<<<1, 32, 0, runtime->stream>>>(
      runtime->frontier, runtime->frontier_authority, runtime->frontier_count,
      runtime->frontier_capacity, runtime->context_state, event,
      DirectIngressAuthority::ordinary);
}

void inject_raw_events(DirectAdultRuntime* runtime, const ActivityEvent* events,
                       std::uint32_t count) {
  if (runtime == nullptr || events == nullptr || count == 0u)
    return;
  if (runtime->ingress_staging == nullptr) {
    // No staging buffer (a runtime built before this field existed): fall back
    // to the one-launch-per-event path so behaviour is never lost.
    for (std::uint32_t i = 0; i < count; ++i)
      inject_raw_event(runtime, events[i]);
    return;
  }
  // Chunked at frontier_capacity because that is the staging buffer's size.
  // Chunk boundaries are invisible to the organism: each chunk resumes the same
  // serial device loop against the same context_state.
  std::uint32_t offset = 0u;
  while (offset < count) {
    const std::uint32_t chunk = std::min(count - offset, runtime->frontier_capacity);
    check_cuda(cudaMemcpy(runtime->ingress_staging, events + offset,
                          chunk * sizeof(ActivityEvent), cudaMemcpyHostToDevice),
               "upload ingress batch");
    // #1304: guarded against underflow -- nothing pushes frontier_launch_bound
    // above frontier_capacity today, but the subtraction was unsigned with no
    // check, so a future caller that violated the invariant would wrap to a
    // huge headroom instead of failing visibly.
    const std::uint32_t headroom = runtime->frontier_launch_bound < runtime->frontier_capacity
                                        ? runtime->frontier_capacity - runtime->frontier_launch_bound
                                        : 0u;
    runtime->frontier_launch_bound += std::min(chunk, headroom);
    append_event_batch_kernel<<<1, 32, 0, runtime->stream>>>(
        runtime->frontier, runtime->frontier_authority, runtime->frontier_count,
        runtime->frontier_capacity, runtime->context_state, runtime->ingress_staging, chunk,
        DirectIngressAuthority::ordinary);
    check_cuda(cudaGetLastError(), "launch append_event_batch_kernel");
    offset += chunk;
  }
}

void inject_bridge_return_event(DirectAdultRuntime* runtime, const ActivityEvent& event,
                                BridgeReturnInjectionGrant) {
  if (runtime == nullptr)
    return;
  if (runtime->frontier_launch_bound < runtime->frontier_capacity)
    ++runtime->frontier_launch_bound;
  append_event_kernel<<<1, 32, 0, runtime->stream>>>(
      runtime->frontier, runtime->frontier_authority, runtime->frontier_count,
      runtime->frontier_capacity, runtime->context_state, event,
      DirectIngressAuthority::bridge_authenticated_return);
}

void mark_bridge_tickets_strict(DirectAdultRuntime* runtime, const BridgeTicketMark* marks,
                                std::uint32_t count) {
  if (runtime == nullptr || marks == nullptr || count == 0u)
    return;
  // Marks are drawn from this step's own drained motor events, so the
  // persistent staging buffer covers every bridge call without touching the
  // allocator mid-stream.  The temporary remains only for an out-of-band caller
  // that exceeds motor_capacity; behaviour is identical either way.
  const bool staged = runtime->bridge_mark_staging != nullptr && count <= runtime->motor_capacity;
  BridgeTicketMark* device_marks = staged ? runtime->bridge_mark_staging : nullptr;
  if (!staged) {
    check_cuda(cudaMalloc(&device_marks, count * sizeof(BridgeTicketMark)),
               "allocate bridge ticket marks");
  }
  check_cuda(
      cudaMemcpy(device_marks, marks, count * sizeof(BridgeTicketMark), cudaMemcpyHostToDevice),
      "upload bridge ticket marks");
  mark_bridge_ticket_strict_kernel<<<grid_for(count), 128, 0, runtime->stream>>>(
      runtime->eligibility_bank, runtime->eligibility_count, runtime->eligibility_capacity,
      runtime->eligibility_bucket_heads, runtime->eligibility_bucket_count, device_marks, count);
  check_cuda(cudaGetLastError(), "launch mark_bridge_ticket_strict_kernel");
  if (!staged)
    check_cuda(cudaFree(device_marks), "free bridge ticket marks");
}

void inject_membrane_boundary(DirectAdultRuntime* runtime) {
  if (runtime == nullptr)
    return;
  reset_resident_context_kernel<<<1, 32, 0, runtime->stream>>>(runtime->context_state);
}

namespace {

// 0X1-#1175: the idle (topology_work == 0) fast path used to issue three
// separate cudaMemsetAsync host dispatches every tick. Fused into one
// kernel; same zero values, same fields.
__global__ void clear_adult_step_idle_state_kernel(std::uint32_t* next_frontier_count,
                                                   std::uint32_t* motor_count,
                                                   AdultCounters* counters) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *next_frontier_count = 0u;
    *motor_count = 0u;
    *counters = AdultCounters{};
    // #1255: written after the zero-init above, and nowhere else -- the sole
    // signal that a step actually took this fast path rather than one that
    // happened to have nothing else to report.
    counters->idle_step_taken = 1u;
  }
}

// 0X1-#1175: the busy per-step path used to issue five separate
// cudaMemsetAsync host dispatches every tick (one of them scaled by
// eligibility_bucket_count). Fused into one kernel launch; same zero/0xff
// values, same sizes.
__global__ void clear_adult_step_busy_state_kernel(std::uint32_t* next_frontier_count,
                                                    std::uint32_t* next_eligibility_count,
                                                    std::uint32_t* next_eligibility_bucket_heads,
                                                    std::uint32_t eligibility_bucket_count,
                                                    std::uint32_t* motor_count,
                                                    AdultCounters* counters) {
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < eligibility_bucket_count;
       i += stride) {
    next_eligibility_bucket_heads[i] = 0xffffffffu;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *next_frontier_count = 0u;
    *next_eligibility_count = 0u;
    *motor_count = 0u;
    *counters = AdultCounters{};
  }
}

// 0X1-1176: `next_frontier_count` is an `atomicAdd` ATTEMPT counter, not a
// residency. Both `append_one_event` and the propagation append do
// `slot = atomicAdd(count, 1)` and store only `if (slot < capacity)`, so on a
// saturated tick the count runs far past the buffer -- ~12280 attempts into 4096
// slots on the reference adult -- and the surplus is discarded by race order.
//
// Left unclamped the lie propagates twice. It becomes the next tick's ingress
// cursor, so ingress starts appending past capacity (that half was repaired in
// 8948240b93 by letting contact displace an endogenous prediction). And it is
// what `AdultStepReceipt::next_frontier_size` reports, so every width, peak and
// occupancy quoted from that field on a saturated adult -- including in several
// of this arc's own receipts -- was about threefold too large.
//
// The surplus is captured before the clamp so it stays countable rather than
// merely disappearing.
__global__ void clamp_frontier_residency_kernel(std::uint32_t* next_frontier_count,
                                                std::uint32_t capacity,
                                                AdultCounters* counters) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t attempted = *next_frontier_count;
    counters->frontier_successors_discarded = attempted > capacity ? attempted - capacity : 0u;
    if (attempted > capacity)
      *next_frontier_count = capacity;
  }
}

}  // namespace

void launch_direct_adult_step(DirectAdultRuntime* runtime) {
  if (runtime == nullptr)
    return;
  // github #1208 item 1.  Every dispatch BELOW is placed on this stream; a
  // default-constructed runtime holds nullptr, which is the legacy default
  // stream, so untouched callers keep byte-identical behaviour.  The calls this
  // step makes into other translation units (topology arena, fabric, tract,
  // packed, checkpoint) still issue to the legacy stream and are therefore what
  // now refuses first under capture.
  const cudaStream_t stream = runtime->stream;
  const std::uint32_t return_credit_block = runtime->return_credit_launch_block == 64u ||
                                                    runtime->return_credit_launch_block == 128u ||
                                                    runtime->return_credit_launch_block == 256u
                                                ? runtime->return_credit_launch_block
                                                : 128u;
  const std::uint32_t frontier_work =
      std::min(runtime->frontier_launch_bound, runtime->frontier_capacity);
  const std::uint32_t eligibility_work =
      std::min(runtime->eligibility_launch_bound, runtime->eligibility_capacity);
  const std::uint32_t topology_work = frontier_work + eligibility_work;

  // #1335: the first topology epoch contains two independent proposal
  // namespaces. Return-credit runs over the full device-resident frontier
  // and writes at ordinal `i`; eligibility retractions must therefore start
  // after the full frontier-capacity prefix rather than after the stale
  // host-tracked frontier_work prefix.
  //
  // proposal_high_water is only a physical touched-prefix optimization
  // inside this lawful address span. It must never be asked to repair an
  // undersized or overlapping ordinal namespace.
  const std::uint32_t return_credit_proposal_span = runtime->frontier_capacity;
  const std::uint32_t eligibility_proposal_span = runtime->eligibility_capacity;
  const std::uint32_t eligibility_proposal_base = return_credit_proposal_span;
  const std::uint32_t preprop_topology_work =
      return_credit_proposal_span + eligibility_proposal_span;

  if (topology_work == 0u) {
    clear_adult_step_idle_state_kernel<<<1, 32, 0, stream>>>(runtime->next_frontier_count,
                                                  runtime->motor_count, runtime->counters);
    check_cuda(cudaGetLastError(), "clear idle adult step state");
    // #1179: a state owner in probation_pending_retract (or rematerialize_
    // pending) must keep retrying every tick, including a run of purely idle
    // ticks that never touch the frontier at all -- this branch never swaps
    // the eligibility banks, so the rebind sub-pass is pointed at the
    // CURRENT bank (not next_eligibility_bank, which is unused/stale here).
    launch_direct_state_owner_step(runtime, 0u, runtime->eligibility_bank,
                                   runtime->eligibility_count);
    std::swap(runtime->frontier, runtime->next_frontier);
    std::swap(runtime->frontier_count, runtime->next_frontier_count);
    runtime->frontier_launch_bound = 0u;
    runtime->eligibility_launch_bound = 0u;
    ++runtime->tick;
    return;
  }

  clear_adult_step_busy_state_kernel<<<grid_for(runtime->eligibility_bucket_count), 128u, 0,
                                     stream>>>(
      runtime->next_frontier_count, runtime->next_eligibility_count,
      runtime->next_eligibility_bucket_heads, runtime->eligibility_bucket_count,
      runtime->motor_count, runtime->counters);
  check_cuda(cudaGetLastError(), "clear busy adult step state");

  reset_direct_topology_proposals(runtime->topology_runtime, preprop_topology_work, stream);
  reset_active_route_credit_kernel<<<grid_for(runtime->eligibility_capacity), 128, 0, stream>>>(
      runtime->brain, runtime->eligibility_bank, runtime->eligibility_count,
      runtime->eligibility_capacity);
  apply_return_credit_kernel<<<grid_for(runtime->frontier_capacity, return_credit_block), return_credit_block, 0,
                              stream>>>(
      runtime->brain, runtime->topology_runtime->view, runtime->frontier,
      runtime->frontier_authority, runtime->frontier_count, runtime->eligibility_bank,
      runtime->eligibility_count, runtime->eligibility_capacity, runtime->eligibility_bucket_heads,
      runtime->eligibility_bucket_count, runtime->tick, runtime->counters,
      runtime->representation->view, runtime->frontier_capacity);
  compact_eligibility_kernel<<<grid_for(runtime->eligibility_capacity), 128, 0, stream>>>(
      runtime->brain, runtime->eligibility_bank, runtime->eligibility_count,
      runtime->eligibility_capacity, runtime->next_eligibility_bank,
      runtime->next_eligibility_count, runtime->eligibility_bucket_heads,
      runtime->next_eligibility_bucket_heads, runtime->eligibility_bucket_count, runtime->tick,
      runtime->counters, runtime->representation->view);
  finalize_active_route_credit_kernel<<<grid_for(runtime->eligibility_capacity), 128, 0, stream>>>(
      runtime->brain, runtime->eligibility_bank, runtime->eligibility_count,
      runtime->eligibility_capacity, runtime->topology_runtime->view, eligibility_proposal_base);
  decide_eligibility_batch_admission_kernel<<<1, 32, 0, stream>>>(
      runtime->next_eligibility_count, runtime->eligibility_capacity,
      runtime->eligibility_batch_admit);
  launch_direct_topology_epoch(&runtime->brain, runtime->topology_runtime, preprop_topology_work,
                               runtime->counters, 128u, stream);
  install_committed_context_routes_kernel<<<grid_for(preprop_topology_work), 128, 0, stream>>>(
      runtime->brain, *runtime->topology_runtime, preprop_topology_work, runtime->counters);

  // #1327/#1236: the sibling of the #1304 grid repair. Both propagate paths now
  // launch across frontier_capacity and self-bound on *frontier_count, and each
  // thread submits its growth proposal at ordinal
  // `i * kMaxImplicitActiveFanout + implicit_ordinal`. Deriving the arena's
  // bound from the host-tracked frontier_work therefore left every proposal from
  // a frontier entry beyond that bound written into the buffer and never
  // scanned by arbitration -- silently dropped, not deferred, so the organism
  // could participate, qualify for materialization and still never grow. The
  // proposal arena is already ALLOCATED for
  // frontier_capacity * kMaxImplicitActiveFanout + eligibility_capacity
  // (create_direct_adult_runtime); the per-step bound must name the same span.
  const std::uint32_t frontier_topology_work =
      runtime->frontier_capacity * kMaxImplicitActiveFanout;
  reset_direct_topology_proposals(runtime->topology_runtime, frontier_topology_work, stream);
  const bool use_fabric = runtime->fabric != nullptr && (runtime->fabric->packed_panel_count > 0u ||
                                                         runtime->fabric->dense_tile_count > 0u ||
                                                         runtime->fabric->tract_lane_count > 0u);

  if (use_fabric) {
    // gh #1236: grid covers frontier_capacity, not frontier_work -- see the
    // kernel's own comment above.
    record_fabric_explicit_eligibility_kernel<<<grid_for(runtime->frontier_capacity), 128, 0,
                                                stream>>>(
        runtime->brain, runtime->fabric->view, runtime->frontier, runtime->frontier_count,
        runtime->next_eligibility_bank, runtime->next_eligibility_count,
        runtime->eligibility_capacity, runtime->next_eligibility_bucket_heads,
        runtime->eligibility_bucket_count, runtime->eligibility_batch_admit, runtime->tick,
        runtime->counters);
    launch_direct_heterogeneous_frontier_step(runtime, frontier_work);
  } else {
    // #1304: the grid must cover the device-resident live frontier, not the
    // host-tracked frontier_launch_bound -- during ingress ramp-up the bound
    // trails the live count for several ticks, and any frontier entry beyond
    // frontier_work threads never receives a thread at all (silently dropped,
    // not deferred). The kernel already self-bounds per-thread against
    // *frontier_count, so launching the full capacity adds no host dispatch
    // calls, only device-side threads that may exit immediately.
    propagate_sparse_frontier_kernel<<<grid_for(runtime->frontier_capacity), 128, 0, stream>>>(
        runtime->brain, runtime->topology_runtime->view, runtime->frontier, runtime->frontier_count,
        runtime->next_frontier, runtime->next_frontier_authority, runtime->next_frontier_count,
        runtime->frontier_capacity, runtime->motor_events, runtime->motor_count,
        runtime->motor_capacity, runtime->next_eligibility_bank, runtime->next_eligibility_count,
        runtime->eligibility_capacity, runtime->next_eligibility_bucket_heads,
        runtime->eligibility_bucket_count, runtime->eligibility_batch_admit, runtime->tick,
        runtime->counters, runtime->representation->view);
  }

  launch_direct_topology_epoch(&runtime->brain, runtime->topology_runtime, frontier_topology_work,
                               runtime->counters, 128u, stream);
  install_committed_context_routes_kernel<<<grid_for(frontier_topology_work), 128, 0, stream>>>(
      runtime->brain, *runtime->topology_runtime, frontier_topology_work, runtime->counters);
  append_committed_growth_eligibility_kernel<<<grid_for(frontier_topology_work), 128, 0,
                                             stream>>>(
      *runtime->topology_runtime, frontier_topology_work, runtime->next_eligibility_bank,
      runtime->next_eligibility_count, runtime->eligibility_capacity,
      runtime->next_eligibility_bucket_heads, runtime->eligibility_bucket_count,
      runtime->eligibility_batch_admit, runtime->brain, runtime->tick, runtime->counters);

  // #1178: Bounded rolling maintenance pass and resource ecology pool synchronization
  if (topology_work > 0u || (runtime->tick & 15u) == 0u) {
    // 0X1-1175: `eligibility_work` is a LAUNCH BOUND -- it ratchets to
    // eligibility_capacity and, by the arithmetic at the end of this function,
    // cannot fall. The parameter it lands in means a live record count, and the
    // pin scan it feeds reports every route hard-pinned when the count exceeds
    // what the bank holds. The bound is still the right grid extent, so it stays;
    // the device-resident live count is passed alongside it and clamps the scan
    // on the device, with no host readback.
    launch_direct_resource_maintenance_step(
        &runtime->brain, runtime->brain.retention_bank, runtime->brain.minimal_retention_bank,
        runtime->eligibility_bank, eligibility_work, runtime->resource_maintenance,
        runtime->tick, 64u, stream, 128u, runtime->eligibility_count);
  }

  // #1179: runs after this tick's topology epoch(s) have committed (so a
  // just-confirmed retract/materialize is visible) and before the frontier/
  // eligibility bank swap (so its rebind pass targets next_eligibility_bank,
  // the set that is about to become "current").
  launch_direct_representation_compiler_step(runtime, frontier_work);
  check_cuda(cudaGetLastError(), "launch direct adult step");

  // 0X1-1176: the count must report residency, not attempts, before it becomes
  // both the next tick's ingress cursor and this tick's reported frontier size.
  clamp_frontier_residency_kernel<<<1, 32, 0, stream>>>(
      runtime->next_frontier_count, runtime->frontier_capacity, runtime->counters);

  std::swap(runtime->frontier, runtime->next_frontier);
  std::swap(runtime->frontier_authority, runtime->next_frontier_authority);
  std::swap(runtime->frontier_count, runtime->next_frontier_count);
  std::swap(runtime->eligibility_bank, runtime->next_eligibility_bank);
  std::swap(runtime->eligibility_count, runtime->next_eligibility_count);
  std::swap(runtime->eligibility_bucket_heads, runtime->next_eligibility_bucket_heads);
  runtime->frontier_launch_bound = frontier_work;
  // The bank this step already covered MUST stay covered. Without the
  // `eligibility_work` term the bound depends only on this step's frontier, so
  // the moment the frontier goes quiet the bound collapses below the live bank:
  // at frontier_work == 0 it reaches 0 while records are still live, the
  // topology_work == 0 fast path above then fires, and the bank is never aged,
  // credited, or expired again -- the bound is a launch bound, not a live count,
  // so nothing can raise it back. Measured as expiry_records 2 -> 1 in
  // cuda_direct_multicontext_eligibility_contract.
  const std::uint64_t eligibility_upper =
      static_cast<std::uint64_t>(eligibility_work) +
      static_cast<std::uint64_t>(frontier_work) * kEligibilityLaunchExpansion;
  runtime->eligibility_launch_bound = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(eligibility_upper, runtime->eligibility_capacity));
  ++runtime->tick;
}

AdultStepReceipt observe_adult_step(const DirectAdultRuntime& runtime) {
  const std::uint32_t* tract_counts =
      runtime.fabric != nullptr ? runtime.fabric->view.tract_bucket_counts : nullptr;
  const std::uint32_t max_delay =
      runtime.fabric != nullptr ? runtime.fabric->view.max_tract_delay : 0u;
  pack_adult_step_receipt_kernel<<<1, 64, 0, runtime.stream>>>(
      runtime.counters, runtime.frontier_count, runtime.motor_count, runtime.eligibility_count,
      tract_counts, max_delay, runtime.step_receipt);
  check_cuda(cudaGetLastError(), "pack step receipt");
  AdultStepReceipt receipt{};
  check_cuda(
      cudaMemcpy(&receipt, runtime.step_receipt, sizeof(AdultStepReceipt), cudaMemcpyDeviceToHost),
      "read adult step receipt");
  return receipt;
}

}  // namespace substrate::direct_adult
