#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTIVATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTIVATION_CUH

#include "hardware_native/direct_adult_device_ops.cuh"

namespace substrate::direct_adult_core {

// Activation authority is resident tissue. Cross-node influence reaches
// node_incoming_excitation through authoritative sparse/delayed routes or dense
// relations. The fixed observer slots may summarize activation but never feed it.
__device__ inline std::int32_t integrate_adult_node_activation(
    DirectNode& node, std::int32_t* incoming_excitation,
    std::int32_t* slow_context_q16, std::uint32_t current_tick,
    std::uint32_t refractory_period, std::int32_t attractor_coupling_gain_q16,
    std::int32_t persistent_bias_ceiling_q16,
    std::int32_t slow_context_ceiling_q16) {
  const std::int32_t in_exc = incoming_excitation != nullptr ? *incoming_excitation : 0;
  if (incoming_excitation != nullptr) *incoming_excitation = 0;

  const std::int32_t slow_ctx = slow_context_q16 != nullptr
                                    ? clamp_q16(*slow_context_q16, 0,
                                                slow_context_ceiling_q16)
                                    : 0;
  if (slow_context_q16 != nullptr) *slow_context_q16 = slow_ctx;

  // The sum is capped at the base activation threshold by default, so stored
  // context cannot fire a node without current route/contact evidence.
  const std::int32_t persistent_bias = clamp_q16(
      (node.attractor_support_q16 / 8) + (slow_ctx / 8), 0,
      persistent_bias_ceiling_q16);
  // Fused internal tissue expresses a longer local timescale through the same
  // integration law in both executors. Motor membrane nodes remain gated by
  // current drive rather than becoming action latches.
  constexpr std::int32_t kDenseBasinRetentionQ16 = 3 * (kQ16One / 4);
  const std::int32_t retained_activation =
      (node.flags & direct_network::kNodeFlagDenseMember) != 0u &&
              (node.flags & direct_network::kNodeFlagMotor) == 0u
          ? mul_q16(node.activation_q16, kDenseBasinRetentionQ16)
          : 0;
  const std::int32_t net_input =
      in_exc + persistent_bias + retained_activation;
  const std::int32_t activation_threshold =
      (kQ16One / 16) + (node.inhibition_q16 / 4);
  const bool refractory = current_tick < node.refractory_until;
  const std::int32_t activation =
      !refractory && net_input > activation_threshold
          ? clamp_q16(net_input, 0, kQ16One)
          : 0;
  node.activation_q16 = activation;

  if (activation > (kQ16One / 4)) {
    node.attractor_support_q16 = clamp_q16(
        node.attractor_support_q16 + (attractor_coupling_gain_q16 / 8), 0,
        kQ16One);
    node.refractory_until = current_tick + refractory_period;
    node.last_endogenous_tick = current_tick;
  } else {
    node.attractor_support_q16 = mul_q16(node.attractor_support_q16, 58982);
  }
  node.activity_ema_q16 = (node.activity_ema_q16 * 31 + activation) / 32;
  return activation;
}

__device__ inline void record_activation_observer(
    AttractorBasinState* observer, std::uint16_t territory_index,
    std::int32_t activation_q16) {
  if (observer != nullptr && activation_q16 > (kQ16One / 4))
    atomicAdd(&observer->active_coalition_nodes[territory_index % kMaxBasins], 1u);
}

__device__ inline void step_activation_observer(AttractorBasinState* observer,
                                                std::uint32_t slot) {
  if (observer == nullptr || slot >= kMaxBasins) return;
  const std::uint32_t active_nodes = observer->active_coalition_nodes[slot];
  observer->active_coalition_nodes[slot] = 0u;
  const std::int32_t active_energy = clamp_q16(
      static_cast<std::int32_t>(active_nodes * 2048u), 0, kQ16One);
  observer->basin_energy_q16[slot] =
      (observer->basin_energy_q16[slot] * 3 + active_energy) / 4;
  if (observer->basin_energy_q16[slot] > (kQ16One / 16)) {
    observer->basin_stability_q16[slot] = clamp_q16(
        observer->basin_stability_q16[slot] + (kQ16One / 8), 0, kQ16One);
    observer->basin_phase[slot] = (observer->basin_phase[slot] + 1u) & 0xffu;
  } else {
    observer->basin_stability_q16[slot] =
        mul_q16(observer->basin_stability_q16[slot], 62259);
  }
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_ACTIVATION_CUH
