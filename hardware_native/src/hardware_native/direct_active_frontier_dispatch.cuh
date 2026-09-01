#ifndef HARDWARE_NATIVE_DIRECT_ACTIVE_FRONTIER_DISPATCH_CUH
#define HARDWARE_NATIVE_DIRECT_ACTIVE_FRONTIER_DISPATCH_CUH

#include <cstdint>

namespace substrate::direct_adult {

constexpr std::uint32_t kDirectActiveDispatchCapacity = 64u;
constexpr std::uint32_t kDirectResidentFrontierOrigin = 0xD1AEC7u;

// Resident causal admission owns this bounded work list. node_capacity is
// solely a range guard and never determines hot-path work.
struct DirectActiveFrontierDispatch {
  std::uint32_t nodes[kDirectActiveDispatchCapacity]{};
  std::uint32_t count = 0u;
  std::uint32_t node_capacity = 0u;
  std::uint32_t origin = 0u;
  std::uint64_t resident_epoch_token = 0u;
};

__device__ inline bool direct_active_frontier_valid(const DirectActiveFrontierDispatch& dispatch) {
  if (dispatch.count == 0u || dispatch.count > kDirectActiveDispatchCapacity ||
      dispatch.node_capacity == 0u || dispatch.origin != kDirectResidentFrontierOrigin ||
      dispatch.resident_epoch_token == 0u)
    return false;
  for (std::uint32_t i = 0u; i < dispatch.count; ++i) {
    if (dispatch.nodes[i] >= dispatch.node_capacity)
      return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (dispatch.nodes[prior] == dispatch.nodes[i])
        return false;
  }
  return true;
}

__device__ inline bool direct_active_frontier_node_for_lane(
    const DirectActiveFrontierDispatch& dispatch, std::uint32_t lane, std::uint32_t* node) {
  if (node == nullptr || lane >= dispatch.count)
    return false;
  *node = dispatch.nodes[lane];
  return true;
}

}  // namespace substrate::direct_adult

#endif
