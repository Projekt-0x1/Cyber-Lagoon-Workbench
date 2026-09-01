// canonical Direct route transport owns its bounded packet working set and the
// host-side dispatch composition point. The device reservation/publication
// protocol remains in direct_adult_device_ops.cuh.

struct AdultWorkingSetCharge {
  std::uint64_t eligibility_units;
  std::uint64_t packet_units;
  std::uint64_t ticket_units;
  std::uint64_t unpooled_bytes;
};

AdultWorkingSetCharge adult_working_set_charge(
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    std::uint32_t action_participant_capacity,
    bool charge_route_opportunity) {
  const std::uint64_t prov_slots =
      static_cast<std::uint64_t>(node_count) * kNodeParticipationAperture;
  const std::uint64_t route_transport_claims =
      std::max(static_cast<std::uint64_t>(route_capacity),
               static_cast<std::uint64_t>(kMaxDelayedSparsePackets)) *
      kRouteTransportProducerWidth;
  const std::uint64_t node_candidate_owners =
      std::max(route_transport_claims, prov_slots);
  AdultWorkingSetCharge charge{};
  charge.eligibility_units = kMaxLiveEligibilityRecords;
  // One pool byte width cannot represent heterogeneous arrays. Keep packet
  // and ticket pool units homogeneous; charge every other allocation exactly
  // once as unpooled bytes.
  charge.packet_units = kMaxIngressQueueSize;
  charge.ticket_units = 2ull * kMaxAsynchronousTickets;
  charge.unpooled_bytes =
      sizeof(IngressRingControl) * 2ull +
      sizeof(ResidentContactEpochReceipt) * kMaxIngressQueueSize +
      sizeof(ResidentActualFrontier) +
      sizeof(ConsequenceIngressEvent) * kMaxAsynchronousTickets +
      sizeof(std::uint32_t) * 5ull + // stop, epoch limit, tick, egress cursors
      sizeof(ResidentAdultEpochSnapshot) +
      sizeof(MotorEvent) * kMaxEgressQueueSize +
      sizeof(std::uint32_t) + // live eligibility count
      sizeof(std::uint64_t) * kEligibilityClaimDirectoryCapacity +
      sizeof(std::uint32_t) *
          (kEligibilityClaimLockCount + kMaxLiveEligibilityRecords) +
      sizeof(std::uint32_t) * 2ull + // ticket count and claim incarnation
      sizeof(std::uint32_t) * kMaxAsynchronousTickets + // ticket locks
      sizeof(ResolvedConsequenceContext) +
      sizeof(direct_network::DirectAffectBodyState) +
      sizeof(AttractorBasinState) +
      sizeof(std::int32_t) * 2ull * node_count +
      sizeof(DelayedSparsePacket) * kMaxDelayedSparsePackets +
      sizeof(std::uint32_t) * (2ull + kMaxDelayedSparsePackets) +
      sizeof(DelayedPacketIdentity) * kDelayedPacketIdentityCapacity +
      sizeof(RouteTransportProposal) * route_capacity +
      sizeof(EligibilityBatchClaim) *
          route_transport_claims +
      (charge_route_opportunity ? sizeof(std::uint64_t) * route_capacity : 0ull) +
      sizeof(NodeCausalParticipation) * 2ull * prov_slots +
      sizeof(std::uint32_t) * 4ull * node_count +
      sizeof(DirectParticipationDescriptor) * kMaxLiveEligibilityRecords +
      sizeof(std::uint32_t) *
          (2ull * route_transport_claims + node_candidate_owners +
           (kMaxLiveEligibilityRecords > kMaxDelayedSparsePackets
                ? kMaxLiveEligibilityRecords
                : kMaxDelayedSparsePackets) +
           kRouteTransportOwnerScratchCapacity +
           1ull) +
      sizeof(std::uint32_t) +
      sizeof(std::uint32_t) *
          ((static_cast<std::uint64_t>(route_capacity) + 31ull) / 32ull) +
      sizeof(AdultCoreMetrics) +
      sizeof(DirectActionParticipationLink) * kMaxAsynchronousTickets *
          static_cast<std::uint64_t>(action_participant_capacity);
  charge.unpooled_bytes +=
      direct_network::resident_development_workspace_bytes(node_count);
  return charge;
}

std::uint64_t adult_eligibility_bytes_per_unit() {
  return sizeof(EligibilityRecord);
}
std::uint64_t adult_ticket_bytes_per_unit() { return sizeof(AsynchronousTicket); }
std::uint64_t adult_packet_bytes_per_unit() { return sizeof(ActivityEvent); }

void refresh_ingress_reclaim_frontier(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->ingress_control == nullptr ||
      runtime->host_ingress_head_snapshot == nullptr) return;
  const cudaStream_t stream = runtime->transport_stream
      ? runtime->transport_stream : runtime->stream;
  if (!runtime->is_persistent_running && runtime->ingress_consumed_event != nullptr)
    cudaStreamWaitEvent(stream, runtime->ingress_consumed_event, 0);
  check_cuda(cudaMemcpyAsync(
                 runtime->host_ingress_head_snapshot,
                 &runtime->ingress_control->consumed_head,
                 sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
             "refresh ingress reclaim frontier");
  check_cuda(cudaStreamSynchronize(stream), "sync ingress reclaim frontier");
  const std::uint32_t head = *runtime->host_ingress_head_snapshot;
  if (runtime->host_ingress_write_tail - head <= kMaxIngressQueueSize)
    runtime->host_ingress_observed_head = head;
}

#include "hardware_native/direct_adult_sensor_port_index.cuh"

std::uint32_t stage_sensory_events(
    DirectAdultRuntime* runtime, const ActivityEvent* events,
    const ResidentContactEpochReceipt* receipts, std::uint32_t count) {
  if (runtime == nullptr || runtime->brain == nullptr || events == nullptr ||
      count == 0u || runtime->host_ingress_contact_staging == nullptr) return 0u;
  // Transport mirrors the device membrane exactly: a born surface admits each
  // contact through its unique matching sensor port, while a port-less
  // runtime accepts only explicit live-node contact.
  if (runtime->host_boundary_ports != nullptr) {
    const auto sensor_role =
        static_cast<std::uint32_t>(direct_network::BoundaryRole::sensor);
    SensorPortEntry sensor_ports[direct_network::kMaxBoundaryPorts];
    std::uint32_t sensor_count = 0u;
    for (std::uint32_t i = 0u; i < runtime->brain->boundary_port_count; ++i)
      if ((runtime->host_boundary_ports[i].role_mask & sensor_role) != 0u)
        sensor_ports[sensor_count++] = {runtime->host_boundary_ports[i].channel,
                                        runtime->host_boundary_ports[i].node, i};
    std::sort(sensor_ports, sensor_ports + sensor_count,
              [](const SensorPortEntry& l, const SensorPortEntry& r) {
                return l.channel < r.channel;
              });
    for (std::uint32_t i = 0u; i < count; ++i)
      if (sorted_sensor_port_index(sensor_ports, sensor_count, events[i].node,
                                   events[i].channel) == kInvalidIndex)
        return 0u;
  } else {
    for (std::uint32_t i = 0u; i < count; ++i)
      if (events[i].node >= runtime->brain->node_count) return 0u;
  }
  std::uint32_t used =
      runtime->host_ingress_write_tail - runtime->host_ingress_observed_head;
  if (used > kMaxIngressQueueSize)
    throw std::runtime_error("stage_sensory_events: invalid ingress ring");
  if (kMaxIngressQueueSize - used < count) {
    refresh_ingress_reclaim_frontier(runtime);
    used = runtime->host_ingress_write_tail - runtime->host_ingress_observed_head;
    if (used > kMaxIngressQueueSize)
      throw std::runtime_error("stage_sensory_events: invalid refreshed ingress ring");
  }
  const std::uint32_t admitted =
      std::min(count, kMaxIngressQueueSize - used);
  runtime->host_ingress_overflow_drops += count - admitted;
  for (std::uint32_t i = 0u; i < admitted; ++i) {
    const std::uint32_t cursor = runtime->host_ingress_write_tail + i;
    const std::uint32_t slot = cursor % kMaxIngressQueueSize;
    runtime->host_ingress_staging[slot] = events[i];
    runtime->host_ingress_contact_staging[slot] =
        receipts != nullptr ? receipts[i] : ResidentContactEpochReceipt{};
  }
  runtime->host_ingress_write_tail += admitted;
  return admitted;
}

void initialize_delayed_packet_transport(DirectAdultRuntime* runtime) {
  auto allocate = [](auto** ptr, std::size_t bytes, const char* name) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), bytes), name);
    check_cuda(cudaMemset(*ptr, 0, bytes), name);
  };
  allocate(&runtime->delayed_packets,
           sizeof(DelayedSparsePacket) * kMaxDelayedSparsePackets,
           "delayed_packets");
  allocate(&runtime->delayed_packet_live_count, sizeof(std::uint32_t),
           "delayed_packet_live_count");
  allocate(&runtime->delayed_packet_free_head, sizeof(std::uint32_t),
           "delayed_packet_free_head");
  allocate(&runtime->delayed_packet_next_free,
           sizeof(std::uint32_t) * kMaxDelayedSparsePackets,
           "delayed_packet_next_free");
  allocate(&runtime->delayed_packet_identities,
           sizeof(DelayedPacketIdentity) * kDelayedPacketIdentityCapacity,
           "delayed_packet_identities");
  const std::uint32_t producer_capacity =
      std::max(runtime->brain->route_capacity, kMaxDelayedSparsePackets);
  runtime->eligibility_batch_capacity =
      producer_capacity * kRouteTransportProducerWidth;
  runtime->route_transport_scan_capacity =
      runtime->eligibility_batch_capacity;
  allocate(&runtime->route_transport_proposals,
           sizeof(RouteTransportProposal) * runtime->brain->route_capacity,
           "route_transport_proposals");
  allocate(&runtime->eligibility_batch_claims,
           sizeof(EligibilityBatchClaim) * runtime->eligibility_batch_capacity,
           "eligibility_batch_claims");
  allocate(&runtime->route_transport_scan_a,
           sizeof(std::uint32_t) * runtime->route_transport_scan_capacity,
           "route_transport_scan_a");
  allocate(&runtime->route_transport_scan_b,
           sizeof(std::uint32_t) * runtime->route_transport_scan_capacity,
           "route_transport_scan_b");
  allocate(&runtime->route_transport_free_slots,
           sizeof(std::uint32_t) *
               (kMaxLiveEligibilityRecords > kMaxDelayedSparsePackets
                    ? kMaxLiveEligibilityRecords
                    : kMaxDelayedSparsePackets),
           "route_transport_free_slots");
  allocate(&runtime->eligibility_batch_owners,
           sizeof(std::uint32_t) * kRouteTransportOwnerScratchCapacity,
           "eligibility_batch_owners");
  const std::uint64_t node_candidate_capacity = std::max(
      static_cast<std::uint64_t>(runtime->eligibility_batch_capacity),
      static_cast<std::uint64_t>(runtime->brain->node_count) *
          kNodeParticipationAperture);
  if (node_candidate_capacity >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
          sizeof(std::uint32_t))
    throw std::overflow_error(
        "node participation candidate-owner allocation overflow");
  allocate(&runtime->node_participation_candidate_owners,
           sizeof(std::uint32_t) *
               static_cast<std::size_t>(node_candidate_capacity),
           "node_participation_candidate_owners");
  allocate(&runtime->route_transport_cursor, sizeof(std::uint32_t),
           "route_transport_cursor");
  const std::uint32_t first_delayed_slot = 0u;
  check_cuda(cudaMemcpy(runtime->delayed_packet_free_head, &first_delayed_slot,
                        sizeof(first_delayed_slot), cudaMemcpyHostToDevice),
             "init delayed_packet_free_head");
  std::vector<std::uint32_t> delayed_next(kMaxDelayedSparsePackets);
  for (std::uint32_t i = 0u; i < kMaxDelayedSparsePackets; ++i)
    delayed_next[i] = i;
  check_cuda(cudaMemcpy(runtime->delayed_packet_next_free, delayed_next.data(),
                        sizeof(std::uint32_t) * delayed_next.size(),
                        cudaMemcpyHostToDevice),
             "init delayed_packet_next_free");
}

void dispatch_deterministic_route_transport(DirectAdultRuntime* runtime,
                                            std::uint32_t block_dim) {
  RouteTransportPhaseView transport{
      runtime->route_transport_proposals,
      runtime->eligibility_batch_claims,
      runtime->eligibility_batch_capacity,
      runtime->route_transport_scan_a,
      runtime->route_transport_scan_b,
      runtime->route_transport_scan_capacity,
      runtime->route_transport_free_slots,
      runtime->eligibility_batch_owners,
      runtime->node_participation_candidate_owners,
      runtime->route_transport_cursor};
  int device = 0;
  int multiprocessors = 0;
  int blocks_per_sm = 0;
  check_cuda(cudaGetDevice(&device), "route transport device");
  check_cuda(cudaDeviceGetAttribute(
                 &multiprocessors, cudaDevAttrMultiProcessorCount, device),
             "route transport multiprocessors");
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks_per_sm, deterministic_route_transport_kernel,
                 static_cast<int>(block_dim), 0u),
             "route transport occupancy");
  const int blocks = multiprocessors * blocks_per_sm;
  if (blocks <= 0)
    throw std::runtime_error("route transport has no cooperative residency");
  const DirectBrain brain = *runtime->brain;
  void* arguments[] = {
      const_cast<DirectBrain*>(&brain),
      &runtime->node_incoming_excitation,
      &runtime->delayed_packets,
      &runtime->delayed_packet_live_count,
      &runtime->delayed_packet_free_head,
      &runtime->delayed_packet_next_free,
      &runtime->delayed_packet_identities,
      &runtime->node_active_participation,
      &runtime->node_next_participation,
      &runtime->node_active_ancestry_incomplete,
      &runtime->node_next_ancestry_incomplete,
      &runtime->participation_staging,
      &runtime->participation_staging_count,
      &runtime->participation_staging_capacity,
      &runtime->eligibility_table,
      &runtime->live_eligibility_count,
      &runtime->eligibility_claim_directory,
      &runtime->eligibility_record_generations,
      &runtime->route_opportunity_incarnations,
      &transport,
      &runtime->current_tick,
      &runtime->config,
      &runtime->device_metrics};
  check_cuda(cudaLaunchCooperativeKernel(
                 reinterpret_cast<void*>(
                     deterministic_route_transport_kernel),
                 blocks, block_dim, arguments, 0u, runtime->stream),
             "launch deterministic route transport");
}

void dispatch_ingress_eligibility_frontier(DirectAdultRuntime* runtime,
                                           std::uint32_t block_dim) {
  int device = 0;
  int multiprocessors = 0;
  int blocks_per_sm = 0;
  check_cuda(cudaGetDevice(&device), "ingress eligibility device");
  check_cuda(cudaDeviceGetAttribute(
                 &multiprocessors, cudaDevAttrMultiProcessorCount, device),
             "ingress eligibility multiprocessors");
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks_per_sm, derive_ingress_eligibility_frontier_kernel,
                 static_cast<int>(block_dim), 0u),
             "ingress eligibility occupancy");
  const int blocks = multiprocessors * blocks_per_sm;
  if (blocks <= 0)
    throw std::runtime_error(
        "ingress eligibility frontier has no cooperative residency");
  void* arguments[] = {
      &runtime->eligibility_table,
      &runtime->eligibility_record_generations,
      &runtime->live_eligibility_count,
      &runtime->current_tick,
      &runtime->route_transport_scan_a,
      &runtime->route_transport_scan_b,
      &runtime->route_transport_free_slots,
      &runtime->device_metrics};
  check_cuda(
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(derive_ingress_eligibility_frontier_kernel),
          blocks, block_dim, arguments, 0u, runtime->stream),
      "launch ingress eligibility frontier");
}
