#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_lived_expression.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#define HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_OPS
#include "hardware_native/direct_adult_resident_motor_trajectory.cuh"
#undef HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_OPS
#define HARDWARE_NATIVE_DEFINE_DENSE_SHATTER_RUNTIME
#include "hardware_native/direct_adult_dense_execution.cuh"
#undef HARDWARE_NATIVE_DEFINE_DENSE_SHATTER_RUNTIME
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_causal_world_model.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"

namespace substrate::direct_adult_core {
using namespace nvcuda;
__global__ void execute_dense_tensor_wmma_kernel(
    const DirectNode* nodes,
    const DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    const std::uint16_t* dense_weights_fp16,
    std::int32_t* node_incoming_excitation,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    std::uint32_t current_tick,
    AdultCoreMetrics* metrics) {
  if (blockIdx.x >= dense_block_count) return;
  __shared__ half shared_weights[kDirectDenseTileElements];
  __shared__ half shared_activation[kDirectDenseTileElements];
  __shared__ float shared_output[kDirectDenseTileElements];
  execute_dense_tensor_block(
      nodes, dense_blocks, blockIdx.x, dense_weights_fp16,
      node_incoming_excitation, node_next_ancestry_incomplete,
      participation_staging, participation_staging_count,
      participation_staging_capacity, current_tick, metrics,
      threadIdx.x & 31u, shared_weights, shared_activation, shared_output);
}

__global__ void integrate_node_activation_kernel(
    DirectNode* nodes,
    std::uint32_t node_count,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    AttractorBasinState* attractor_state,
    std::uint32_t current_tick,
    std::uint32_t refractory_period,
    std::int32_t attractor_coupling_gain_q16,
    std::int32_t persistent_bias_ceiling_q16,
    std::int32_t slow_context_ceiling_q16) {
  const std::uint32_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_idx >= node_count) return;

  DirectNode& node = nodes[node_idx];
  const std::int32_t activation = integrate_adult_node_activation(
      node, node_incoming_excitation + node_idx,
      node_slow_context_q16 + node_idx, current_tick, refractory_period,
      attractor_coupling_gain_q16, persistent_bias_ceiling_q16,
      slow_context_ceiling_q16);
  record_activation_observer(attractor_state, node.territory_index, activation);
}

__global__ void decay_node_slow_context_kernel(
    std::int32_t* node_slow_context_q16, std::uint32_t node_count,
    std::int32_t decay_q16) {
  const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node < node_count)
    decay_resident_transient_trace_q16(node_slow_context_q16 + node, decay_q16);
}

__global__ void step_attractor_basins_kernel(
    AttractorBasinState* attractor_state,
    std::uint32_t basin_count) {
  const std::uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= basin_count || b >= kMaxBasins) return;

  step_activation_observer(attractor_state, b);
}

__global__ void derive_affect_body_state_kernel(
    direct_network::DirectAffectBodyState* state,
    const AsynchronousTicket* tickets, ResidentDevelopmentState* development) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      tickets == nullptr || development == nullptr)
    return;
  const direct_network::DirectExactHistoryHotPage& history =
      development->exact_history;
  direct_network::affect_derive_from_ledgers(
      state, tickets, kMaxAsynchronousTickets, history.records,
      history.committed_slots, 0u, kInvalidIndex);
}

__global__ void derive_wanting_liking_state_kernel(
    direct_network::ResidentWantingLikingProfileV1* state,
    const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* links, std::uint32_t link_count,
    const direct_network::DirectAffectBodyState* affect,
    const direct_network::ResidentRecipeCell* recipes,
    std::uint32_t recipe_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      actions == nullptr || links == nullptr)
    return;
  *state = direct_network::observe_resident_wanting_liking(
      actions, kMaxAsynchronousTickets, links, link_count, affect, recipes,
      recipe_count);
}

__global__ void derive_resident_causal_world_model_kernel(
    direct_network::DirectCausalWorldModel* model,
    const AsynchronousTicket* tickets, const std::uint32_t* ticket_count,
    ResidentDevelopmentState* development,
    const direct_network::ResidentPostbirthConstructorState* constructor_state,
    const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || model == nullptr ||
      tickets == nullptr || development == nullptr)
    return;
  (void)ticket_count;
  const auto& history = development->exact_history;
  direct_network::derive_causal_world_relations_from_action_participation(
      model, tickets, kMaxAsynchronousTickets, history.records,
      history.committed_slots, constructor_state, actions, action_links,
      action_participant_capacity);
}


#include "hardware_native/direct_adult_motor_affect_helpers.cuh"
#define HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL
#include "hardware_native/direct_adult_resident_motor_trajectory.cuh"
#undef HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL


}  // namespace substrate::direct_adult_core
