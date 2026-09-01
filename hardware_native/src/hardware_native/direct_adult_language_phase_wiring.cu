#ifdef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#undef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#endif

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_language_expression_opportunity.cuh"
#include "hardware_native/direct_adult_language_phase_wiring.cuh"
#include "hardware_native/direct_adult_resident_language_runtime_abi.cuh"

namespace substrate::direct_adult_core {

void launch_resident_language_assimilation_phase(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->brain == nullptr ||
      runtime->brain->development == nullptr)
    return;
  direct_network::launch_direct_resident_language_assimilation(
      runtime->resident_language, &runtime->brain->development->exact_history,
      runtime->stream);
}

void launch_resident_language_motor_finalize_phase(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->brain == nullptr) return;
  launch_finalize_resident_language_motor(
      runtime->resident_language, *runtime->brain, runtime->actual_frontier,
      runtime->egress_queue, runtime->egress_head, runtime->egress_tail,
      runtime->ticket_table, runtime->action_occurrences,
      runtime->action_participation_links, runtime->brain->development,
      runtime->efference_ring, runtime->efference_head, runtime->efference_tail,
      runtime->config.route_efference_copies, runtime->current_tick,
      runtime->stream);
}

}  // namespace substrate::direct_adult_core
