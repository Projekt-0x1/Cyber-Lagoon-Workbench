// Patch 0005: __global__ kernel declarations for one life tick. Fanout
// kernels are defined in bcc32_network_activity.cu; the credit
// classification kernel is defined in bcc32_network_credit.cu (a distinct
// causal-credit concern from ordinary activity fanout, per the patch
// program's own file split). Declared together here so
// run_one_activity_tick() can launch all of them via CUDA's normal
// separable-compilation model.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_KERNELS_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_KERNELS_CUH

#include "hardware_native/bcc32_network_activity.cuh"

namespace substrate::bcc32::network_recipe {

__global__ void clear_activity_next_frontier_kernel(std::uint32_t* next_frontier_count);

// Fans out every event currently in `rail_in` through its node's live
// children into `rail_out`, preserving origin (the rail a shadow event
// propagates into is itself a shadow next-frontier, never the actual one,
// and vice versa -- distinct physical rails, not just a tag). Accumulates
// real per-child actual_traffic/shadow_traffic on the target NetworkNode,
// and -- only for shadow-origin events -- refreshes predicted_word at the
// child so a later actual_world_return recontact has something to compare
// against.
__global__ void fanout_activity_kernel(NetworkNode* nodes, std::uint32_t node_capacity,
                                        const NetworkActivityEvent* frontier_in,
                                        const std::uint32_t* frontier_count_in,
                                        NetworkActivityEvent* frontier_out,
                                        std::uint32_t* frontier_count_out,
                                        std::uint32_t out_capacity, SiteWord* predicted_word,
                                        std::uint32_t tick);

// Classifies every actual_world_return event in the actual rail against
// that node's outstanding predicted_word: match (credit, revision++),
// violation, or omission (no prediction existed). Consumes the prediction
// (resets predicted_word to kNoPrediction) on match or violation -- an
// omission leaves predicted_word untouched since there was nothing to
// consume. Writes per-event outcome counts into `report` (a 3-element
// array: [matches, violations, omissions], atomically accumulated).
__global__ void classify_and_credit_kernel(NetworkNode* nodes,
                                            const NetworkActivityEvent* actual_frontier_in,
                                            const std::uint32_t* actual_frontier_count_in,
                                            SiteWord* predicted_word, std::uint32_t* report);

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_KERNELS_CUH
