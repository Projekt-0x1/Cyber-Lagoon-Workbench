// Patch 0005: fanout kernel and host orchestration for one life tick. See
// bcc32_network_activity.cuh for the shared types and
// bcc32_network_credit.cu for the delayed-credit classification kernel.

#include "hardware_native/bcc32_network_activity_kernels.cuh"

#include <vector>

namespace substrate::bcc32::network_recipe {

__global__ void clear_activity_next_frontier_kernel(std::uint32_t* next_frontier_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *next_frontier_count = 0u;
}

__global__ void fanout_activity_kernel(NetworkNode* nodes, std::uint32_t node_capacity,
                                        const NetworkActivityEvent* frontier_in,
                                        const std::uint32_t* frontier_count_in,
                                        NetworkActivityEvent* frontier_out,
                                        std::uint32_t* frontier_count_out,
                                        std::uint32_t out_capacity, SiteWord* predicted_word,
                                        std::uint32_t tick) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *frontier_count_in) return;

  const NetworkActivityEvent event = frontier_in[i];
  if (event.node >= node_capacity) return;
  NetworkNode& node = nodes[event.node];
  const bool actual = is_actual_origin(event.origin);

  for (std::uint32_t c = 0; c < kMaximumChildren; ++c) {
    const std::uint32_t child = node.child[c];
    if (child == kInvalidNodeIndex || child >= node_capacity) continue;

    if (actual) {
      atomicAdd(&nodes[child].actual_traffic, 1u);
    } else {
      atomicAdd(&nodes[child].shadow_traffic, 1u);
      // Only a shadow event refreshes the child's outstanding prediction --
      // an actual event recontacting a node is what gets *classified*
      // against a prediction (bcc32_network_credit.cu), never what writes
      // one; conflating the two would let an actual event "predict itself".
      predicted_word[child] = event.word ^ node.edge_chemistry[c];
    }

    const std::uint32_t out_i = atomicAdd(frontier_count_out, 1u);
    if (out_i < out_capacity) {
      NetworkActivityEvent& out = frontier_out[out_i];
      out.node = child;
      out.word = event.word ^ node.edge_chemistry[c];
      out.origin = event.origin;
      out.producer = event.node;
      out.source_route = event.source_route;
      out.parent_route = event.node;
      out.horizon = event.horizon + 1;
    }
  }
  (void)tick;
}

namespace {
std::uint32_t grid_size(std::uint32_t work_items, std::uint32_t block_size) {
  return (work_items + block_size - 1) / block_size;
}

void run_one_rail(NetworkNode* nodes, std::uint32_t node_capacity, ActivityRail& rail,
                   SiteWord* predicted_word, std::uint32_t tick, std::uint32_t block_size,
                   std::uint32_t& host_in_count) {
  clear_activity_next_frontier_kernel<<<1, 1>>>(rail.next_frontier_count);
  cudaMemcpy(&host_in_count, rail.frontier_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
  if (host_in_count > 0) {
    fanout_activity_kernel<<<grid_size(host_in_count, block_size), block_size>>>(
        nodes, node_capacity, rail.frontier, rail.frontier_count, rail.next_frontier,
        rail.next_frontier_count, rail.capacity, predicted_word, tick);
  }
}

void swap_rail(ActivityRail& rail) {
  NetworkActivityEvent* frontier_swap = rail.frontier;
  rail.frontier = rail.next_frontier;
  rail.next_frontier = frontier_swap;
  std::uint32_t* count_swap = rail.frontier_count;
  rail.frontier_count = rail.next_frontier_count;
  rail.next_frontier_count = count_swap;
}
}  // namespace

ActivityTickReport run_one_activity_tick(NetworkActivityDeviceState& state,
                                          std::uint32_t block_size) {
  ActivityTickReport report{};

  run_one_rail(state.nodes, state.node_capacity, state.actual, state.predicted_word, state.tick,
               block_size, report.actual_events_processed);
  run_one_rail(state.nodes, state.node_capacity, state.shadow, state.predicted_word, state.tick,
               block_size, report.shadow_events_processed);

  // Classify actual_world_return events from *this tick's incoming* actual
  // rail (before the fanout above overwrote it) against their nodes'
  // outstanding predictions. Order matters: classification reads the
  // predicted_word a *prior* shadow tick wrote, so it must run using the
  // same frontier the fanout above just consumed, not the fanned-out
  // children -- credit is about the event's own arrival node, not where it
  // propagates to next.
  std::uint32_t credit_report[3] = {0, 0, 0};
  std::uint32_t* device_credit_report = nullptr;
  cudaMalloc(&device_credit_report, sizeof(credit_report));
  cudaMemset(device_credit_report, 0, sizeof(credit_report));
  if (report.actual_events_processed > 0) {
    classify_and_credit_kernel<<<grid_size(report.actual_events_processed, block_size),
                                  block_size>>>(state.nodes, state.actual.frontier,
                                                 state.actual.frontier_count, state.predicted_word,
                                                 device_credit_report);
  }
  cudaMemcpy(credit_report, device_credit_report, sizeof(credit_report), cudaMemcpyDeviceToHost);
  cudaFree(device_credit_report);
  report.matches = credit_report[0];
  report.violations = credit_report[1];
  report.omissions = credit_report[2];

  swap_rail(state.actual);
  swap_rail(state.shadow);

  cudaMemcpy(&report.new_actual_frontier_size, state.actual.frontier_count, sizeof(std::uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&report.new_shadow_frontier_size, state.shadow.frontier_count, sizeof(std::uint32_t),
             cudaMemcpyDeviceToHost);

  ++state.tick;
  return report;
}

}  // namespace substrate::bcc32::network_recipe
