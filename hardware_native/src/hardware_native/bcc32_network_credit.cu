// Patch 0005: delayed causal credit. An actual_world_return event
// recontacting a node that carried an outstanding shadow prediction is
// classified into exactly one of three distinct physical aftermath states
// -- match, violation, omission -- never collapsed into a single
// right/wrong bit. Only a match awards credit (NetworkNode.revision++,
// the same field patch 0003 already reserved, reused rather than
// duplicated). This is the mechanism that lets a returned world
// consequence recarve exactly the route that produced it, per the patch
// program's "actual/shadow recurrent activity graph" section.

#include "hardware_native/bcc32_network_activity_kernels.cuh"

namespace substrate::bcc32::network_recipe {

__global__ void classify_and_credit_kernel(NetworkNode* nodes,
                                            const NetworkActivityEvent* actual_frontier_in,
                                            const std::uint32_t* actual_frontier_count_in,
                                            SiteWord* predicted_word, std::uint32_t* report) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *actual_frontier_count_in) return;

  const NetworkActivityEvent event = actual_frontier_in[i];
  if (event.origin != NetworkEventOrigin::actual_world_return) return;

  NetworkNode& node = nodes[event.node];
  const SiteWord prediction = predicted_word[event.node];

  if (prediction == kNoPrediction) {
    atomicOr(&node.flags, kNodeFlagOmission);
    atomicAdd(&report[2], 1u);  // omissions
    return;
  }

  if (prediction == event.word) {
    atomicOr(&node.flags, kNodeFlagMatch);
    atomicAdd(&node.revision, 1u);  // delayed causal credit
    atomicAdd(&report[0], 1u);      // matches
  } else {
    atomicOr(&node.flags, kNodeFlagViolation);
    atomicAdd(&report[1], 1u);  // violations
  }

  // The prediction is consumed either way -- a confirmed or violated
  // prediction does not linger to be reused by a later, unrelated actual
  // event; renewed contact must earn a fresh prediction.
  predicted_word[event.node] = kNoPrediction;
}

}  // namespace substrate::bcc32::network_recipe
