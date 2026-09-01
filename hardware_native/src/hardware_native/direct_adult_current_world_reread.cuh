#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CURRENT_WORLD_REREAD_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CURRENT_WORLD_REREAD_CUH

// f.current_world_reread (#1023). Later action and later report must READ the
// CURRENT world through the causal relation's freshest verified return --
// never replay a historical answer, never substitute the internally predicted
// majority. The reread therefore fails closed unless actual return evidence
// at least as fresh as the caller's required horizon exists.

#include <cstdint>

#include "hardware_native/direct_adult_causal_world_model.cuh"

namespace substrate::direct_network {

// Read the current world value one relation carries. The call demands actual
// reread evidence: the relation must exist, its newest verified return must
// be at least as fresh as `required_freshness_tick`, and that return must be
// non-empty. The majority prediction is deliberately not consulted -- an
// internally assumed answer cannot stand in for reading the world.
__host__ __device__ inline bool reread_current_world_value(
    const DirectCausalWorldModel& model, std::uint32_t action_value,
    std::uint32_t root_channel, std::uint32_t required_freshness_tick,
    std::uint32_t* current_value) {
  if (current_value == nullptr) return false;
  const std::int32_t index =
      causal_model_find_relation(model, action_value, root_channel);
  if (index < 0) return false;
  const DirectCausalRelation& relation = model.relations[index];
  if (relation.current_outcome_tick < required_freshness_tick ||
      relation.current_outcome_value == 0u)
    return false;
  *current_value = relation.current_outcome_value;
  return true;
}

}  // namespace substrate::direct_network

#endif
