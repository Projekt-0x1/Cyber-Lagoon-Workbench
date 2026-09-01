// Patch 0004: host orchestration for one developmental tick. See
// bcc32_network_life_function.cuh for the shared types and this landing's
// documented scope, and bcc32_network_construction_kernels.cu for the
// kernels themselves.

#include "hardware_native/bcc32_network_construction_kernels.cuh"

namespace substrate::bcc32::network_recipe {

namespace {
std::uint32_t grid_size(std::uint32_t work_items, std::uint32_t block_size) {
  return (work_items + block_size - 1) / block_size;
}

__global__ void clear_tick_report_kernel(TickReport* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *report = TickReport{};
}

__global__ void finalize_tick_report_kernel(TickReport* report,
                                            const std::uint32_t* frontier_count,
                                            std::uint32_t frontier_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint32_t count = *frontier_count;
  report->new_frontier_size = count < frontier_capacity ? count : frontier_capacity;
}
}  // namespace

TickReport run_one_tick(LifeFunctionDeviceState& state, TickReport* device_report,
                         std::uint32_t block_size) {
  // clear_claims_kernel only needs to run before a gestation's very first
  // tick. commit_claims_kernel (bcc32_network_construction_kernels.cu) now
  // resets every claim slot it processes back to its default/unclaimed
  // state immediately after reading it, so the table this tick's commit
  // leaves behind is already clean for the next tick's
  // evaluate_and_claim_kernel writes -- see that kernel's comment. Paying a
  // standalone full kClaimTableSize-wide clear sweep on *every* tick
  // regardless of the live frontier's size was exactly GitHub #1167's named
  // "cost must scale with |Q_t|, never with allocated capacity |S_t|"
  // violation; this removes it for every tick after the first. This relies
  // on state.tick==0 meaning "this state's claim table has never been
  // committed into yet" -- true for every current caller (each sets
  // state.tick=0 once at setup, before its first run_one_tick() call); a
  // future checkpoint-resume caller that reuses a mid-gestation claims
  // buffer under a fresh LifeFunctionDeviceState would need to either
  // preserve that invariant or force one clear_claims_kernel call itself.
  if (state.tick == 0) {
    clear_claims_kernel<<<grid_size(kClaimTableSize, block_size), block_size>>>(state.claims);
  }
  clear_next_frontier_kernel<<<1, 1>>>(state.next_frontier_count);
  clear_tick_report_kernel<<<1, 1>>>(device_report);

  if (state.frontier_size > 0) {
    evaluate_and_claim_kernel<<<grid_size(state.frontier_size, block_size), block_size>>>(
        state, device_report);
  }

  commit_claims_kernel<<<grid_size(kClaimTableSize, block_size), block_size>>>(state,
                                                                               device_report);
  TickReport report{};
  cudaMemcpy(&report, device_report, sizeof(report), cudaMemcpyDeviceToHost);

  // Advance to the constructed frontier: swap buffers and counters so the
  // caller's next run_one_tick() call reads exactly this tick's new nodes.
  FrontierEntry* frontier_swap = state.frontier;
  state.frontier = state.next_frontier;
  state.next_frontier = frontier_swap;

  std::uint32_t* count_swap = state.frontier_count;
  state.frontier_count = state.next_frontier_count;
  state.next_frontier_count = count_swap;
  state.frontier_size = report.new_frontier_size;

  ++state.tick;

  return report;
}

CapturedTickReport run_captured_ticks(LifeFunctionDeviceState& state, TickReport* device_report,
                                        std::uint32_t tick_count, std::uint32_t block_size) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

  clear_tick_report_kernel<<<1, 1, 0, stream>>>(device_report);
  if (state.tick == 0) {
    clear_claims_kernel<<<grid_size(kClaimTableSize, block_size), block_size, 0, stream>>>(
        state.claims);
  }

  FrontierEntry* buffers[2] = {state.frontier, state.next_frontier};
  std::uint32_t* counts[2] = {state.frontier_count, state.next_frontier_count};
  const std::uint32_t final_index = tick_count % 2;

  const std::uint32_t claim_grid = grid_size(kClaimTableSize, block_size);
  // Worst-case bound: capture cannot synchronously read the real frontier
  // count back to the host between ticks, so every tick's
  // evaluate_and_claim_kernel is launched wide enough to cover every
  // possible live node -- see the header comment for why this is correct
  // (the kernel's own bounds check) but not free (idle threads).
  const std::uint32_t eval_grid = grid_size(state.node_capacity, block_size);

  for (std::uint32_t t = 0; t < tick_count; ++t) {
    LifeFunctionDeviceState tick_state = state;
    tick_state.frontier = buffers[t % 2];
    tick_state.frontier_count = counts[t % 2];
    tick_state.next_frontier = buffers[(t + 1) % 2];
    tick_state.next_frontier_count = counts[(t + 1) % 2];
    tick_state.tick = state.tick + t;

    clear_next_frontier_kernel<<<1, 1, 0, stream>>>(tick_state.next_frontier_count);
    evaluate_and_claim_kernel<<<eval_grid, block_size, 0, stream>>>(tick_state, device_report);
    commit_claims_kernel<<<claim_grid, block_size, 0, stream>>>(tick_state, device_report);
  }
  finalize_tick_report_kernel<<<1, 1, 0, stream>>>(device_report, counts[final_index],
                                                   state.node_capacity);

  cudaGraph_t graph;
  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t graph_exec;
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

  cudaGraphLaunch(graph_exec, stream);
  cudaStreamSynchronize(stream);

  TickReport host_totals{};
  cudaMemcpy(&host_totals, device_report, sizeof(host_totals), cudaMemcpyDeviceToHost);

  // Leave `state` exactly as `tick_count` sequential run_one_tick() calls
  // would have: frontier/next_frontier point at the newest/oldest buffer,
  // and tick has advanced by tick_count.
  state.frontier = buffers[final_index];
  state.frontier_count = counts[final_index];
  state.next_frontier = buffers[(final_index + 1) % 2];
  state.next_frontier_count = counts[(final_index + 1) % 2];
  state.frontier_size = host_totals.new_frontier_size;
  state.tick += tick_count;

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  CapturedTickReport report{};
  report.committed_total = host_totals.committed;
  report.too_many_parents_total = host_totals.too_many_parents;
  report.child_slots_exhausted_total = host_totals.child_slots_exhausted;
  report.page_full_total = host_totals.page_full;
  report.matter_exhausted_total = host_totals.matter_exhausted;
  report.final_frontier_size = host_totals.new_frontier_size;
  report.unsupported_opcode_total = host_totals.unsupported_opcode;
  return report;
}

}  // namespace substrate::bcc32::network_recipe
