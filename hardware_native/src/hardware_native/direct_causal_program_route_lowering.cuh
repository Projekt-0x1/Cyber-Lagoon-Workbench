#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_ROUTE_LOWERING_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_ROUTE_LOWERING_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_causal_program_executor.cuh"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_CAUSAL_ROUTE_HD __host__ __device__
#else
#define DIRECT_CAUSAL_ROUTE_HD
#endif

struct RouteLoweringReceipt {
  std::uint32_t route_index;
  std::uint32_t source_node;
  std::uint32_t target_node;
  std::uint32_t due_tick;
  std::uint32_t context_signature;
  std::uint32_t step_index;
  std::uint64_t route_incarnation;
  std::uint64_t parent_eligibility_ref;
  std::int32_t conductance_q16;
  std::int32_t route_eligibility_q16;
  std::uint32_t admitted;
};
static_assert(std::is_standard_layout_v<RouteLoweringReceipt> &&
              std::is_trivial_v<RouteLoweringReceipt>);

template <typename ParentEligibilityT>
DIRECT_CAUSAL_ROUTE_HD inline RouteLoweringReceipt resolve_program_step_route(
    const ProgramExecutionState& execution, const ParentEligibilityT& parent,
    const substrate::direct_network::DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    std::uint64_t current_parent_eligibility_ref, std::uint32_t current_tick) {
  using substrate::direct_network::kRouteFlagActive;
  RouteLoweringReceipt out{};
  if (execution.active == 0u || execution.completed != 0u ||
      execution.cursor >= execution.active_program.step_count ||
      execution.active_program.initiation_parent_eligibility_ref == 0u ||
      current_parent_eligibility_ref == 0u ||
      parent.live == 0u || parent.expiry_tick < current_tick ||
      parent.lineage_expiry_tick < current_tick || parent.eligibility_q16 <= 0 ||
      routes == nullptr || route_incarnations == nullptr || route_capacity == 0u ||
      current_tick < execution.start_tick)
    return out;

  const ProgramStep& step = execution.active_program.steps[execution.cursor];
  const std::uint32_t due_tick = execution.start_tick + step.due_offset;
  if (due_tick < current_tick) return out;

  std::uint32_t match = route_capacity;
  for (std::uint32_t i = 0u; i < route_capacity; ++i) {
    const auto& route = routes[i];
    if ((route.flags & kRouteFlagActive) == 0u ||
        route.source != parent.target_node || route.target != step.node ||
        route_incarnations[i] == 0u ||
        route.delay > due_tick - current_tick ||
        current_tick + route.delay != due_tick ||
        route.conductance_q16 <= 0)
      continue;
    if (match != route_capacity) return RouteLoweringReceipt{};
    match = i;
  }
  if (match >= route_capacity) return out;
  const auto& route = routes[match];
  out.route_index = match;
  out.source_node = route.source;
  out.target_node = route.target;
  out.due_tick = due_tick;
  out.context_signature = route.eligibility_context;
  out.step_index = execution.cursor;
  out.route_incarnation = route_incarnations[match];
  out.parent_eligibility_ref = current_parent_eligibility_ref;
  out.conductance_q16 = route.conductance_q16;
  out.route_eligibility_q16 = route.eligibility_q16;
  out.admitted = 1u;
  return out;
}

#undef DIRECT_CAUSAL_ROUTE_HD

}  // namespace substrate::direct_causal_program

#endif
