#include <vector>
#include <cub/cub.cuh>
#include "hardware_native/direct_execution_fabric.cuh"

namespace substrate::direct_adult {

__global__ void enqueue_tract_packet_kernel(
    DirectExecutionFabricDeviceView fabric,
    DirectTractPacket packet,
    std::uint32_t stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const std::uint32_t bucket = packet.due_tick % (fabric.max_tract_delay + 1u);
  const std::uint32_t slot = atomicAdd(&fabric.tract_bucket_counts[bucket], 1u);
  if (slot < stride) {
    fabric.tract_ring_packets[bucket * stride + slot] = packet;
  }
}

__global__ void gather_tract_sort_keys_kernel(
    const DirectTractPacket* bucket_packets,
    std::uint32_t bucket_count,
    std::uint64_t* keys_out,
    std::uint32_t* values_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bucket_count)
    return;
  keys_out[i] = bucket_packets[i].sequence_key;
  values_out[i] = i;
}

__global__ void deliver_sorted_tract_packets_kernel(
    const DirectTractPacket* bucket_packets,
    const std::uint32_t* sorted_indices,
    std::uint32_t bucket_count,
    ActivityEvent* staged_events,
    std::uint32_t* staged_event_valid,
    std::uint32_t base_offset,
    AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bucket_count)
    return;
  const std::uint32_t idx = sorted_indices[i];
  const DirectTractPacket pkt = bucket_packets[idx];
  staged_events[base_offset + i] = pkt.event;
  staged_event_valid[base_offset + i] = 1u;
  if (counters != nullptr)
    atomicAdd(&counters->tract_deliveries, 1u);
}

__global__ void clear_tract_bucket_count_kernel(
    std::uint32_t* bucket_counts,
    std::uint32_t bucket) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    bucket_counts[bucket] = 0u;
  }
}

// #1327: `direct_adult_legacy_oracle` is built with CUDA_RESOLVE_DEVICE_SYMBOLS
// ON, so nvlink device-links the library in isolation and drops device code
// nothing inside it reaches. When #1236 quarantined organism-driven tract
// admission (Assay 1 of cuda_direct_tract_delay_fabric_contract) this kernel
// lost its last in-library launcher and was garbage-collected out of the device
// image -- `nm` still showed the host stub, `cuobjdump -elf` no longer showed
// the kernel, and launching it returned cudaErrorSymbolNotFound (500) while
// writing nothing. This launcher keeps the ring's fill half reachable under
// explicit host authority while adult cognition stays quarantined out of it.
// #1208: this is also the ONLY place a tract packet can enter the ring, so it is
// where the ring stops being provably empty. The flag is sticky and never
// cleared -- it can only over-approximate, so the drain can never skip a packet
// that exists. Should a device-side producer ever return, it must set this too,
// or it will be enqueueing into a ring the step is entitled to ignore.
void enqueue_direct_tract_packet(DirectExecutionFabricRuntime* fabric,
                                 const DirectTractPacket& packet) {
  if (fabric == nullptr)
    return;
  fabric->tract_ring_ever_written = true;
  enqueue_tract_packet_kernel<<<1, 1>>>(fabric->view, packet, fabric->node_count);
}

std::uint32_t get_pending_tract_packet_count(
    const DirectExecutionFabricRuntime& fabric) {
  std::vector<std::uint32_t> counts(fabric.max_tract_delay + 1u, 0u);
  cudaMemcpy(counts.data(), fabric.view.tract_bucket_counts,
             sizeof(std::uint32_t) * counts.size(), cudaMemcpyDeviceToHost);
  std::uint32_t total = 0u;
  for (std::uint32_t c : counts)
    total += c;
  return total;
}

}  // namespace substrate::direct_adult
