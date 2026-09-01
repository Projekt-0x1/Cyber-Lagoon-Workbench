namespace {
constexpr std::uint64_t kNullArenaOffset = ~std::uint64_t{0};

struct CheckpointBufferView {
  void* pointer;
  std::size_t bytes;
};

std::array<CheckpointBufferView, kDirectAdultCheckpointBufferCount>
checkpoint_buffer_views(DirectAdultRuntime& runtime) {
  const std::size_t node_count = runtime.brain->node_count;
  const std::size_t route_capacity = runtime.brain->route_capacity;
  const std::size_t participation_slots =
      node_count * kNodeParticipationAperture;
  const std::size_t action_links =
      kMaxAsynchronousTickets * runtime.config.action_participant_capacity;
  const std::size_t pin_words = (route_capacity + 31u) / 32u;
  return {{{runtime.ingress_queue, sizeof(ActivityEvent) * kMaxIngressQueueSize},
           {runtime.ingress_contact_credentials,
            sizeof(ResidentContactEpochReceipt) * kMaxIngressQueueSize},
           {runtime.actual_frontier, sizeof(ResidentActualFrontier)},
           {runtime.mismatch_omission, sizeof(ResidentMismatchOmissionRuntime)},
           {runtime.ingress_control, sizeof(IngressRingControl)},
           {runtime.consequence_queue,
            sizeof(ConsequenceIngressEvent) * kMaxAsynchronousTickets},
           {runtime.consequence_control, sizeof(IngressRingControl)},
           {runtime.egress_queue, sizeof(MotorEvent) * kMaxEgressQueueSize},
           {runtime.egress_head, sizeof(std::uint32_t)},
           {runtime.egress_tail, sizeof(std::uint32_t)},
           {runtime.eligibility_table,
            sizeof(EligibilityRecord) * kMaxLiveEligibilityRecords},
           {runtime.live_eligibility_count, sizeof(std::uint32_t)},
           {runtime.eligibility_claim_directory,
            sizeof(std::uint64_t) * kEligibilityClaimDirectoryCapacity},
           {runtime.eligibility_claim_locks,
            sizeof(std::uint32_t) * kEligibilityClaimLockCount},
           {runtime.eligibility_record_generations,
            sizeof(std::uint32_t) * kMaxLiveEligibilityRecords},
           {runtime.ticket_table, sizeof(AsynchronousTicket) * kMaxAsynchronousTickets},
           {runtime.ticket_count, sizeof(std::uint32_t)},
           {runtime.ticket_table_locks,
            sizeof(std::uint32_t) * kMaxAsynchronousTickets},
           {runtime.claim_incarnation_counter, sizeof(std::uint32_t)},
           {runtime.action_occurrences,
            sizeof(DirectActionOccurrence) * kMaxAsynchronousTickets},
           {runtime.action_participation_links,
            sizeof(DirectActionParticipationLink) * action_links},
           {runtime.resident_motor_trajectory,
            sizeof(ResidentPublicMotorTrajectory)},
           {runtime.resolved_consequence_ctx,
            sizeof(ResolvedConsequenceContext) +
                sizeof(direct_network::DirectAffectBodyState) +
                sizeof(direct_network::ResidentWantingLikingProfileV1) +
                sizeof(direct_network::DirectCausalWorldModel)},
           {runtime.attractor_state, sizeof(AttractorBasinState)},
           {runtime.node_incoming_excitation, sizeof(std::int32_t) * node_count},
           {runtime.node_slow_context_q16, sizeof(std::int32_t) * node_count},
           {runtime.delayed_packets,
            sizeof(DelayedSparsePacket) * kMaxDelayedSparsePackets},
           {runtime.delayed_packet_live_count, sizeof(std::uint32_t)},
           {runtime.delayed_packet_free_head, sizeof(std::uint32_t)},
           {runtime.delayed_packet_next_free,
            sizeof(std::uint32_t) * kMaxDelayedSparsePackets},
           {runtime.delayed_packet_identities,
            sizeof(DelayedPacketIdentity) * kDelayedPacketIdentityCapacity},
           {runtime.route_opportunity_incarnations,
            sizeof(std::uint64_t) * route_capacity},
           {runtime.node_active_participation,
            sizeof(NodeCausalParticipation) * participation_slots},
           {runtime.node_next_participation,
            sizeof(NodeCausalParticipation) * participation_slots},
           {runtime.node_active_participation_locks,
            sizeof(std::uint32_t) * node_count},
           {runtime.node_next_participation_locks,
            sizeof(std::uint32_t) * node_count},
           {runtime.node_active_ancestry_incomplete,
            sizeof(std::uint32_t) * node_count},
           {runtime.node_next_ancestry_incomplete,
            sizeof(std::uint32_t) * node_count},
           {runtime.participation_staging,
            sizeof(DirectParticipationDescriptor) * runtime.participation_staging_capacity},
           {runtime.participation_staging_count, sizeof(std::uint32_t)},
           {runtime.route_causal_pin_bits, sizeof(std::uint32_t) * pin_words},
           {runtime.device_metrics, sizeof(AdultCoreMetrics)},
           {runtime.device_stop_flag, sizeof(std::uint32_t)},
           {runtime.device_resident_tick, sizeof(std::uint32_t)},
           {runtime.device_epoch_limit, sizeof(std::uint32_t)},
           {runtime.device_epoch_snapshot, sizeof(ResidentAdultEpochSnapshot)},
           {runtime.resident_development.counters,
            sizeof(direct_network::ResidentDevelopmentCounters)},
           {runtime.resident_development.reserve_snapshot,
            sizeof(std::uint64_t)},
           {runtime.route_transport_cursor, sizeof(std::uint32_t)},
           {runtime.resident_language,
            direct_network::direct_resident_language_runtime_storage_bytes()},
           {runtime.action_control_runtime,
            direct_adult_action_control_runtime_storage_bytes()}}};
}

template <typename T>
std::uint64_t arena_offset(const T* pointer, const DirectBrain& brain) {
  if (pointer == nullptr) return kNullArenaOffset;
  const auto base = reinterpret_cast<std::uintptr_t>(brain.arena);
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address < base || address >= base + brain.arena_bytes) {
    throw std::runtime_error("capture_direct_adult_checkpoint: arena pointer escaped");
  }
  return address - base;
}

template <typename T>
void restore_arena_pointer(T*& pointer, void* arena, std::uint64_t arena_bytes,
                           std::uint64_t offset, std::uint64_t count) {
  if (offset == kNullArenaOffset) {
    if (count != 0u) {
      throw std::invalid_argument(
          "restore_direct_adult_checkpoint: missing arena view");
    }
    pointer = nullptr;
    return;
  }
  if (offset > arena_bytes || count > (arena_bytes - offset) / sizeof(T)) {
    throw std::invalid_argument("restore_direct_adult_checkpoint: invalid arena offset");
  }
  pointer = reinterpret_cast<T*>(static_cast<std::byte*>(arena) + offset);
}

void synchronize_checkpoint_boundary(const DirectAdultRuntime& runtime) {
  if (runtime.is_persistent_running ||
      runtime.execution_authority != AdultExecutionAuthority::host_stepped) {
    throw std::runtime_error(
        "capture_direct_adult_checkpoint: adult is not at a host-stepped boundary");
  }
  if (runtime.host_ingress_write_tail != runtime.host_ingress_publish_tail ||
      runtime.host_consequence_observed_head !=
          runtime.host_consequence_write_tail) {
    throw std::runtime_error(
        "capture_direct_adult_checkpoint: unpublished host staging");
  }
  check_cuda(cudaStreamSynchronize(runtime.stream), "checkpoint adult stream");
  check_cuda(cudaStreamSynchronize(runtime.transport_stream),
             "checkpoint transport stream");
  check_cuda(cudaStreamSynchronize(runtime.persistent_stream),
             "checkpoint persistent stream");
}

void clear_brain_pointers(DirectBrain* brain) {
  brain->arena = nullptr; brain->nodes = nullptr; brain->routes = nullptr;
  brain->route_incarnations = nullptr; brain->route_opportunity_incarnations = nullptr;
  brain->route_delay_law_indices = nullptr; brain->route_mature_delays = nullptr;
  brain->route_delay_law_incarnations = nullptr; brain->dense_blocks = nullptr;
  brain->dense_weight_fp16_bits = nullptr; brain->boundary_ports = nullptr; brain->territory_ancestry = nullptr;
  brain->resident_fields = nullptr; brain->resident_field_ranges = nullptr;
  brain->resident_field_indices = nullptr; brain->resident_rules = nullptr;
  brain->resident_tract_delay_laws = nullptr; brain->recipe_cells = nullptr;
  brain->recipe_edges = nullptr; brain->recipe_ranges = nullptr;
  brain->recipe_indices = nullptr; brain->development = nullptr; brain->construction_fronts = nullptr; brain->construction_front_count = nullptr; brain->construction_front_generation_by_node = nullptr; brain->postbirth_derivations = nullptr; brain->postbirth_constructor = nullptr;
  brain->retention_bank = nullptr; brain->resource_ecology = nullptr;
}

}  // namespace

namespace {

void digest_u32_le(substrate::direct_network::detail::DirectSha256State* state,
                   std::uint32_t value) {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8u),
      static_cast<std::uint8_t>(value >> 16u), static_cast<std::uint8_t>(value >> 24u)};
  state->update(bytes, sizeof(bytes));
}

void digest_u64_le(substrate::direct_network::detail::DirectSha256State* state,
                   std::uint64_t value) {
  std::uint8_t bytes[8];
  for (std::uint32_t i = 0u; i < 8u; ++i)
    bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
  state->update(bytes, sizeof(bytes));
}

void digest_bytes(substrate::direct_network::detail::DirectSha256State* state,
                  const void* bytes, std::size_t size) {
  digest_u64_le(state, size);
  if (size != 0u) state->update(bytes, size);
}

template <typename T>
void digest_trivial(substrate::direct_network::detail::DirectSha256State* state,
                    const T& value) {
  state->update(&value, sizeof(T));
}

}  // namespace

substrate::direct_network::DirectSha256Address
direct_adult_checkpoint_payload_digest(
    const DirectAdultCheckpoint& checkpoint) {
  static constexpr char kDomain[] = "0x1-direct-adult-checkpoint";
  substrate::direct_network::detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  digest_u32_le(&state, checkpoint.format_version);
  // brain and config arrive from value-initialized hosts, so inter-field
  // padding is zeroed and the raw bytes are stable across captures.
  digest_trivial(&state, checkpoint.brain);
  if (checkpoint.has_resource_ecology) {
    digest_trivial(&state, checkpoint.resource_ecology);
  }
  digest_trivial(&state, checkpoint.arena_pointer_offsets);
  digest_bytes(&state, checkpoint.arena.data(), checkpoint.arena.size());
  for (const auto& buffer : checkpoint.device_buffers) {
    digest_bytes(&state, buffer.data(), buffer.size());
  }
  digest_bytes(&state, checkpoint.host_ingress_staging.data(),
               checkpoint.host_ingress_staging.size() * sizeof(ActivityEvent));
  digest_bytes(&state, checkpoint.host_ingress_contact_staging.data(),
               checkpoint.host_ingress_contact_staging.size() *
                   sizeof(ResidentContactEpochReceipt));
  digest_bytes(&state, checkpoint.host_consequence_staging.data(),
               checkpoint.host_consequence_staging.size() *
                   sizeof(ConsequenceIngressEvent));
  digest_trivial(&state, checkpoint.config);
  digest_u64_le(&state, checkpoint.resident_development_epochs);
  digest_u32_le(&state, checkpoint.current_tick);
  digest_u32_le(&state, checkpoint.participation_staging_capacity);
  digest_u32_le(&state, checkpoint.host_ingress_write_tail);
  digest_u32_le(&state, checkpoint.host_ingress_publish_tail);
  digest_u32_le(&state, checkpoint.host_ingress_observed_head);
  digest_u32_le(&state, checkpoint.host_ingress_dispatched_tail);
  digest_u64_le(&state, checkpoint.host_ingress_overflow_drops);
  digest_u64_le(&state, checkpoint.host_ingress_protocol_faults);
  digest_u32_le(&state, checkpoint.host_consequence_write_tail);
  digest_u32_le(&state, checkpoint.host_consequence_publish_tail);
  digest_u32_le(&state, checkpoint.host_consequence_observed_head);
  digest_u64_le(&state, checkpoint.host_consequence_overflow_drops);
  digest_u64_le(&state, checkpoint.host_consequence_protocol_faults);
  digest_u32_le(&state, checkpoint.host_ingress_head_snapshot);
  digest_u32_le(&state, checkpoint.host_ingress_publish_slot);
  digest_u32_le(&state, checkpoint.host_consequence_head_snapshot);
  digest_u32_le(&state, checkpoint.host_consequence_publish_slot);
  state.update(&checkpoint.has_resource_ecology, 1u);
  return state.finish();
}

IngressRingControl canonicalize_checkpoint_consequence(DirectAdultCheckpoint&, const DirectAdultRuntime&);

DirectAdultCheckpoint capture_direct_adult_checkpoint(
    const DirectAdultRuntime& runtime) {
  if (runtime.brain == nullptr || runtime.brain->arena == nullptr ||
      runtime.brain->arena_bytes == 0u) {
    throw std::invalid_argument("capture_direct_adult_checkpoint: missing adult arena");
  }
  synchronize_checkpoint_boundary(runtime);

  DirectAdultCheckpoint checkpoint{};
  checkpoint.brain = *runtime.brain;
  checkpoint.arena_pointer_offsets = {
      arena_offset(runtime.brain->nodes, *runtime.brain),
      arena_offset(runtime.brain->routes, *runtime.brain),
      arena_offset(runtime.brain->route_incarnations, *runtime.brain),
      arena_offset(runtime.brain->route_opportunity_incarnations, *runtime.brain),
      arena_offset(runtime.brain->route_delay_law_indices, *runtime.brain),
      arena_offset(runtime.brain->route_mature_delays, *runtime.brain),
      arena_offset(runtime.brain->route_delay_law_incarnations, *runtime.brain),
      arena_offset(runtime.brain->dense_blocks, *runtime.brain),
      arena_offset(runtime.brain->dense_weight_fp16_bits, *runtime.brain),
      arena_offset(runtime.brain->boundary_ports, *runtime.brain),
      arena_offset(runtime.brain->resident_fields, *runtime.brain),
      arena_offset(runtime.brain->resident_rules, *runtime.brain),
      arena_offset(runtime.brain->resident_tract_delay_laws, *runtime.brain),
      arena_offset(runtime.brain->recipe_cells, *runtime.brain),
      arena_offset(runtime.brain->recipe_edges, *runtime.brain),
      arena_offset(runtime.brain->recipe_ranges, *runtime.brain),
      arena_offset(runtime.brain->recipe_indices, *runtime.brain),
      arena_offset(runtime.brain->development, *runtime.brain),
      arena_offset(runtime.brain->territory_ancestry, *runtime.brain),
      arena_offset(runtime.brain->resident_field_ranges, *runtime.brain),
      arena_offset(runtime.brain->resident_field_indices, *runtime.brain), arena_offset(runtime.brain->construction_fronts, *runtime.brain), arena_offset(runtime.brain->construction_front_count, *runtime.brain), arena_offset(runtime.brain->construction_front_generation_by_node, *runtime.brain), arena_offset(runtime.brain->postbirth_derivations, *runtime.brain), arena_offset(runtime.brain->postbirth_constructor, *runtime.brain), arena_offset(runtime.brain->retention_bank, *runtime.brain)};
  checkpoint.arena.resize(runtime.brain->arena_bytes);
  check_cuda(cudaMemcpy(checkpoint.arena.data(), runtime.brain->arena,
                        checkpoint.arena.size(), cudaMemcpyDeviceToHost),
             "checkpoint arena");
  checkpoint.has_resource_ecology = runtime.brain->resource_ecology != nullptr;
  if (checkpoint.has_resource_ecology) {
    check_cuda(cudaMemcpy(&checkpoint.resource_ecology, runtime.brain->resource_ecology,
                          sizeof(checkpoint.resource_ecology), cudaMemcpyDeviceToHost),
               "checkpoint resource ecology");
  }
  clear_brain_pointers(&checkpoint.brain);

  DirectAdultRuntime& mutable_runtime =
      const_cast<DirectAdultRuntime&>(runtime);
  const auto buffers = checkpoint_buffer_views(mutable_runtime);
  for (std::size_t index = 0u; index < buffers.size(); ++index) {
    if (buffers[index].bytes == 0u) continue;
    if (buffers[index].pointer == nullptr) {
      throw std::runtime_error("capture_direct_adult_checkpoint: missing runtime buffer");
    }
    checkpoint.device_buffers[index].resize(buffers[index].bytes);
    check_cuda(cudaMemcpy(checkpoint.device_buffers[index].data(),
                          buffers[index].pointer, buffers[index].bytes,
                          cudaMemcpyDeviceToHost),
               "checkpoint runtime buffer");
  }

  const auto consequence_control = canonicalize_checkpoint_consequence(checkpoint, runtime);
  checkpoint.host_ingress_staging.assign(
      runtime.host_ingress_staging,
      runtime.host_ingress_staging + kMaxIngressQueueSize);
  checkpoint.host_ingress_contact_staging.assign(
      runtime.host_ingress_contact_staging,
      runtime.host_ingress_contact_staging + kMaxIngressQueueSize);
  checkpoint.host_consequence_staging.assign(
      runtime.host_consequence_staging,
      runtime.host_consequence_staging + kMaxAsynchronousTickets);
  checkpoint.config = runtime.config;
  checkpoint.resident_development_epochs = runtime.resident_development_epochs;
  checkpoint.current_tick = runtime.current_tick;
  checkpoint.participation_staging_capacity = runtime.participation_staging_capacity;
  checkpoint.host_ingress_write_tail = runtime.host_ingress_write_tail;
  checkpoint.host_ingress_publish_tail = runtime.host_ingress_publish_tail;
  checkpoint.host_ingress_observed_head = runtime.host_ingress_observed_head;
  checkpoint.host_ingress_dispatched_tail = runtime.host_ingress_dispatched_tail;
  checkpoint.host_ingress_overflow_drops = runtime.host_ingress_overflow_drops;
  checkpoint.host_ingress_protocol_faults = runtime.host_ingress_protocol_faults;
  checkpoint.host_consequence_write_tail = runtime.host_consequence_write_tail;
  checkpoint.host_consequence_publish_tail = consequence_control.published_tail;
  checkpoint.host_consequence_observed_head = consequence_control.consumed_head;
  checkpoint.host_consequence_overflow_drops = runtime.host_consequence_overflow_drops;
  checkpoint.host_consequence_protocol_faults = runtime.host_consequence_protocol_faults;
  checkpoint.host_ingress_head_snapshot = *runtime.host_ingress_head_snapshot;
  checkpoint.host_ingress_publish_slot = *runtime.host_ingress_publish_slot_pinned;
  checkpoint.host_consequence_head_snapshot = consequence_control.consumed_head;
  checkpoint.host_consequence_publish_slot = consequence_control.published_tail;
  checkpoint.payload_sha256 = direct_adult_checkpoint_payload_digest(checkpoint);
  return checkpoint;
}

namespace {

template <typename T>
std::vector<T> checkpoint_buffer_copy(const DirectAdultCheckpoint& checkpoint,
                                      std::size_t index,
                                      std::size_t count,
                                      const char* name) {
  if (index >= checkpoint.device_buffers.size())
    throw std::invalid_argument(std::string(
        "restore_direct_adult_checkpoint: invalid ") + name + " buffer index");
  const auto& bytes = checkpoint.device_buffers[index];
  if (bytes.size() != sizeof(T) * count)
    throw std::invalid_argument(std::string(
        "restore_direct_adult_checkpoint: invalid ") + name + " buffer");
  std::vector<T> values(count);
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

struct RestoredTransportIndexes {
  std::uint32_t eligibility_live_count = 0u;
  std::vector<std::uint64_t> eligibility_directory;
  std::vector<std::uint32_t> eligibility_locks;
  std::vector<DelayedSparsePacket> packets;
  std::uint32_t packet_live_count = 0u;
  std::uint32_t packet_free_head = kInvalidIndex;
  std::vector<std::uint32_t> packet_next_free;
  std::vector<DelayedPacketIdentity> packet_identities;
};

RestoredTransportIndexes validate_and_rebuild_transport_indexes(
    const DirectAdultCheckpoint& checkpoint) {
  RestoredTransportIndexes rebuilt{};
  const auto records = checkpoint_buffer_copy<EligibilityRecord>(
      checkpoint, kDirectAdultCheckpointEligibilityTableBuffer,
      kMaxLiveEligibilityRecords, "eligibility table");
  const auto generations = checkpoint_buffer_copy<std::uint32_t>(
      checkpoint, kDirectAdultCheckpointEligibilityGenerationsBuffer,
      kMaxLiveEligibilityRecords, "eligibility generations");
  rebuilt.eligibility_directory.assign(
      kEligibilityClaimDirectoryCapacity, 0u);
  rebuilt.eligibility_locks.assign(kEligibilityClaimLockCount, 0u);
  for (std::uint32_t slot = 0u; slot < kMaxLiveEligibilityRecords; ++slot) {
    const EligibilityRecord& record = records[slot];
    if (record.live > 1u)
      throw std::invalid_argument(
          "restore_direct_adult_checkpoint: invalid eligibility live state");
    if (record.live == 0u || record.expiry_tick < checkpoint.current_tick ||
        record.lineage_expiry_tick < checkpoint.current_tick)
      continue;
    const char* invalid_eligibility = nullptr;
    if (generations[slot] == 0u || generations[slot] == ~0u)
      invalid_eligibility = "generation";
    else if (record.ticket_id == 0u || record.ticket_id == kInvalidTicket)
      invalid_eligibility = "ticket";
    else if (record.source_node >= checkpoint.brain.node_count ||
             record.target_node >= checkpoint.brain.node_count)
      invalid_eligibility = "node";
    else if (record.route_index >= checkpoint.brain.route_capacity)
      invalid_eligibility = "route";
    else if (record.authority == DirectParticipationAuthority::none)
      invalid_eligibility = "authority";
    else if (record.ancestry_depth == 0u ||
             record.ancestry_depth > kMaxProvenanceSlotsPerNode)
      invalid_eligibility = "ancestry depth";
    if (invalid_eligibility != nullptr)
      throw std::invalid_argument(
          std::string("restore_direct_adult_checkpoint: invalid active eligibility ") +
          invalid_eligibility);
    if (record.parent_eligibility_ref == 0u) {
      if (record.ancestry_depth != 1u)
        throw std::invalid_argument(
            "restore_direct_adult_checkpoint: invalid root eligibility depth");
    } else {
      const std::uint32_t encoded_parent =
          static_cast<std::uint32_t>(record.parent_eligibility_ref);
      const std::uint32_t parent_generation =
          static_cast<std::uint32_t>(record.parent_eligibility_ref >> 32u);
      if (encoded_parent == 0u || parent_generation == 0u)
        throw std::invalid_argument(
            "restore_direct_adult_checkpoint: invalid eligibility parent ref");
      const std::uint32_t parent_slot = encoded_parent - 1u;
      if (parent_slot >= kMaxLiveEligibilityRecords ||
          generations[parent_slot] != parent_generation ||
          records[parent_slot].live == 0u ||
          records[parent_slot].expiry_tick < checkpoint.current_tick ||
          records[parent_slot].lineage_expiry_tick < checkpoint.current_tick ||
          records[parent_slot].ticket_id != record.ticket_id ||
          records[parent_slot].target_node != record.source_node ||
          records[parent_slot].claim_incarnation != record.claim_incarnation ||
          records[parent_slot].authority != record.authority ||
          records[parent_slot].authority_incarnation !=
              record.authority_incarnation ||
          records[parent_slot].ancestry_depth + 1u !=
              record.ancestry_depth ||
          records[parent_slot].lineage_expiry_tick <
              record.lineage_expiry_tick)
        throw std::invalid_argument(
            "restore_direct_adult_checkpoint: stale eligibility parent ref");
    }

    const DirectParticipationDescriptor claim =
        eligibility_record_claim(record);
    const std::uint32_t start =
        static_cast<std::uint32_t>(eligibility_claim_hash(claim)) &
        (kEligibilityClaimDirectoryCapacity - 1u);
    bool indexed = false;
    for (std::uint32_t probe = 0u; probe < kEligibilityClaimMaxProbes;
         ++probe) {
      const std::uint32_t directory_slot =
          (start + probe) & (kEligibilityClaimDirectoryCapacity - 1u);
      const std::uint64_t prior =
          rebuilt.eligibility_directory[directory_slot];
      if (prior == 0u) {
        rebuilt.eligibility_directory[directory_slot] =
            (static_cast<std::uint64_t>(generations[slot]) << 32u) |
            (static_cast<std::uint64_t>(slot) + 1u);
        indexed = true;
        break;
      }
      const std::uint32_t prior_slot =
          static_cast<std::uint32_t>(prior) - 1u;
      if (prior_slot < kMaxLiveEligibilityRecords &&
          eligibility_record_identity_matches(record, records[prior_slot]))
        throw std::invalid_argument(
            "restore_direct_adult_checkpoint: duplicate eligibility identity");
    }
    if (!indexed)
      throw std::invalid_argument(
          "restore_direct_adult_checkpoint: unresolved eligibility index");
    ++rebuilt.eligibility_live_count;
  }

  rebuilt.packets = checkpoint_buffer_copy<DelayedSparsePacket>(
      checkpoint, kDirectAdultCheckpointDelayedPacketsBuffer,
      kMaxDelayedSparsePackets, "delayed packets");
  rebuilt.packet_next_free.assign(kMaxDelayedSparsePackets, kInvalidIndex);
  rebuilt.packet_identities.assign(kDelayedPacketIdentityCapacity,
                                   DelayedPacketIdentity{});
  for (std::uint32_t slot = 0u; slot < kMaxDelayedSparsePackets; ++slot) {
    DelayedSparsePacket& packet = rebuilt.packets[slot];
    if (packet.live ==
        static_cast<std::uint32_t>(DelayedPacketSlotState::free))
      continue;
    const char* invalid_packet = nullptr;
    if (packet.live !=
        static_cast<std::uint32_t>(DelayedPacketSlotState::published))
      invalid_packet = "state";
    else if (packet.due_tick <= checkpoint.current_tick ||
             packet.due_tick - checkpoint.current_tick >
                 kMaxPhysicalRouteDelayTicks)
      invalid_packet = "due tick";
    else if (packet.source_node >= checkpoint.brain.node_count ||
             packet.target_node >= checkpoint.brain.node_count)
      invalid_packet = "node";
    else if (packet.route_index >= checkpoint.brain.route_capacity)
      invalid_packet = "route";
    else if (packet.claim_count > kMaxProvenanceSlotsPerNode)
      invalid_packet = "claim count";
    else if (!delayed_packet_causal_payload_structural(
                 packet, checkpoint.brain.node_count,
                 checkpoint.brain.route_capacity))
      invalid_packet = "causal payload";
    if (invalid_packet != nullptr)
      throw std::invalid_argument(
          std::string("restore_direct_adult_checkpoint: invalid delayed packet ") +
          invalid_packet);
    const std::uint32_t start = delayed_packet_identity_hash(
        packet.route_index, packet.due_tick, packet.route_incarnation);
    bool indexed = false;
    for (std::uint32_t probe = 0u; probe < kDelayedPacketIdentityMaxProbes;
         ++probe) {
      const std::uint32_t identity_slot =
          (start + probe) & (kDelayedPacketIdentityCapacity - 1u);
      DelayedPacketIdentity& identity =
          rebuilt.packet_identities[identity_slot];
      if (identity.state ==
          static_cast<std::uint32_t>(DelayedPacketIdentityState::free)) {
        identity = DelayedPacketIdentity{
            static_cast<std::uint32_t>(
                DelayedPacketIdentityState::published),
            slot + 1u, packet.route_index, packet.due_tick,
            packet.route_incarnation};
        packet.reserved0 = identity_slot;
        indexed = true;
        break;
      }
      if (delayed_packet_identity_matches(
              identity, packet.route_index, packet.due_tick,
              packet.route_incarnation)) {
        const std::uint32_t prior_slot = identity.packet_index_plus_one - 1u;
        const bool same_payload =
            prior_slot < kMaxDelayedSparsePackets &&
            delayed_packet_payload_matches(rebuilt.packets[prior_slot], packet);
        throw std::invalid_argument(
            same_payload
                ? "restore_direct_adult_checkpoint: duplicate delayed packet"
                : "restore_direct_adult_checkpoint: duplicate delayed packet payload mismatch");
      }
    }
    if (!indexed)
      throw std::invalid_argument(
          "restore_direct_adult_checkpoint: unresolved delayed packet index");
    rebuilt.packet_next_free[rebuilt.packet_live_count++] = slot;
  }
  std::uint32_t free_rank = rebuilt.packet_live_count;
  for (std::uint32_t slot = 0u; slot < kMaxDelayedSparsePackets; ++slot)
    if (rebuilt.packets[slot].live ==
        static_cast<std::uint32_t>(DelayedPacketSlotState::free))
      rebuilt.packet_next_free[free_rank++] = slot;
  rebuilt.packet_free_head =
      rebuilt.packet_live_count < kMaxDelayedSparsePackets
          ? rebuilt.packet_next_free[rebuilt.packet_live_count]
          : kInvalidIndex;
  return rebuilt;
}

void install_restored_transport_indexes(
    DirectAdultRuntime& runtime, const RestoredTransportIndexes& rebuilt) {
  check_cuda(cudaMemcpy(runtime.live_eligibility_count,
                        &rebuilt.eligibility_live_count,
                        sizeof(rebuilt.eligibility_live_count),
                        cudaMemcpyHostToDevice),
             "restore rebuilt eligibility live count");
  check_cuda(cudaMemcpy(runtime.eligibility_claim_directory,
                        rebuilt.eligibility_directory.data(),
                        sizeof(std::uint64_t) *
                            rebuilt.eligibility_directory.size(),
                        cudaMemcpyHostToDevice),
             "restore rebuilt eligibility directory");
  check_cuda(cudaMemcpy(runtime.eligibility_claim_locks,
                        rebuilt.eligibility_locks.data(),
                        sizeof(std::uint32_t) * rebuilt.eligibility_locks.size(),
                        cudaMemcpyHostToDevice),
             "restore cleared eligibility locks");
  check_cuda(cudaMemcpy(runtime.delayed_packets, rebuilt.packets.data(),
                        sizeof(DelayedSparsePacket) * rebuilt.packets.size(),
                        cudaMemcpyHostToDevice),
             "restore validated delayed packets");
  check_cuda(cudaMemcpy(runtime.delayed_packet_live_count,
                        &rebuilt.packet_live_count,
                        sizeof(rebuilt.packet_live_count),
                        cudaMemcpyHostToDevice),
             "restore rebuilt delayed live count");
  check_cuda(cudaMemcpy(runtime.delayed_packet_free_head,
                        &rebuilt.packet_free_head,
                        sizeof(rebuilt.packet_free_head),
                        cudaMemcpyHostToDevice),
             "restore rebuilt delayed free head");
  check_cuda(cudaMemcpy(runtime.delayed_packet_next_free,
                        rebuilt.packet_next_free.data(),
                        sizeof(std::uint32_t) * rebuilt.packet_next_free.size(),
                        cudaMemcpyHostToDevice),
             "restore rebuilt delayed free list");
  check_cuda(cudaMemcpy(runtime.delayed_packet_identities,
                        rebuilt.packet_identities.data(),
                        sizeof(DelayedPacketIdentity) *
                            rebuilt.packet_identities.size(),
                        cudaMemcpyHostToDevice),
             "restore rebuilt delayed packet index");
}

}  // namespace

DirectAdultRuntime* restore_direct_adult_checkpoint(
    const DirectAdultCheckpoint& checkpoint, DirectBrain* out_brain) {
  if (checkpoint.format_version != kDirectAdultCheckpointVersion) {
    throw std::invalid_argument(
        "restore_direct_adult_checkpoint: unsupported checkpoint version");
  }
  if (direct_adult_checkpoint_payload_digest(checkpoint) !=
      checkpoint.payload_sha256) {
    throw std::invalid_argument(
        "restore_direct_adult_checkpoint: checkpoint payload digest mismatch");
  }
  if (out_brain == nullptr || out_brain->arena != nullptr ||
      checkpoint.brain.arena_bytes == 0u ||
      checkpoint.brain.arena_bytes != checkpoint.arena.size() ||
      checkpoint.participation_staging_capacity != kMaxLiveEligibilityRecords ||
      checkpoint.host_ingress_staging.size() != kMaxIngressQueueSize ||
      checkpoint.host_ingress_contact_staging.size() != kMaxIngressQueueSize ||
      checkpoint.host_consequence_staging.size() != kMaxAsynchronousTickets) {
    throw std::invalid_argument("restore_direct_adult_checkpoint: invalid checkpoint shape");
  }
  const RestoredTransportIndexes rebuilt_transport =
      validate_and_rebuild_transport_indexes(checkpoint);

  *out_brain = checkpoint.brain;
  check_cuda(cudaMalloc(&out_brain->arena, checkpoint.arena.size()),
             "restore arena allocation");
  try {
    check_cuda(cudaMemcpy(out_brain->arena, checkpoint.arena.data(),
                          checkpoint.arena.size(), cudaMemcpyHostToDevice),
               "restore arena");
    const auto& offsets = checkpoint.arena_pointer_offsets;
    auto restore = [&](auto& pointer, std::size_t index, std::uint64_t count) {
      restore_arena_pointer(pointer, out_brain->arena, out_brain->arena_bytes,
                            offsets[index], count);
    };
    const std::uint64_t delay_route_count = out_brain->resident_tract_delay_law_count
                                                ? out_brain->route_capacity : 0u;
    restore(out_brain->nodes, 0u, out_brain->node_count);
    restore(out_brain->routes, 1u, out_brain->route_capacity);
    restore(out_brain->route_incarnations, 2u, out_brain->route_capacity);
    restore(out_brain->route_opportunity_incarnations, 3u,
            offsets[3] == kNullArenaOffset ? 0u : out_brain->route_capacity);
    restore(out_brain->route_delay_law_indices, 4u, delay_route_count);
    restore(out_brain->route_mature_delays, 5u, delay_route_count);
    restore(out_brain->route_delay_law_incarnations, 6u, delay_route_count);
    restore(out_brain->dense_blocks, 7u, out_brain->dense_block_count);
    restore(out_brain->dense_weight_fp16_bits, 8u, out_brain->dense_weight_count);
    restore(out_brain->boundary_ports, 9u, out_brain->boundary_port_count);
    restore(out_brain->resident_fields, 10u, out_brain->resident_field_count);
    restore(out_brain->resident_rules, 11u, out_brain->resident_rule_count);
    restore(out_brain->resident_tract_delay_laws, 12u,
            static_cast<std::uint64_t>(out_brain->resident_tract_delay_law_count) * 5u);
    restore(out_brain->recipe_cells, 13u, out_brain->recipe_cell_count);
    restore(out_brain->recipe_edges, 14u, out_brain->recipe_edge_count);
    restore(out_brain->recipe_ranges, 15u, out_brain->recipe_range_count);
    restore(out_brain->recipe_indices, 16u, out_brain->recipe_index_count);
    restore(out_brain->development, 17u, 1u);
    restore(out_brain->territory_ancestry, 18u, out_brain->territory_ancestry_count);
    restore(out_brain->resident_field_ranges, 19u,
            out_brain->resident_field_range_count);
    restore(out_brain->resident_field_indices, 20u, out_brain->resident_field_index_count); restore(out_brain->construction_fronts, 21u, offsets[21] == kNullArenaOffset ? 0u : out_brain->construction_front_capacity);
    restore(out_brain->construction_front_count, 22u, offsets[22] == kNullArenaOffset ? 0u : 1u); restore(out_brain->construction_front_generation_by_node, 23u, offsets[23] == kNullArenaOffset ? 0u : out_brain->node_count); restore(out_brain->postbirth_derivations, 24u, offsets[24] == kNullArenaOffset ? 0u : direct_network::kResidentPostbirthRecipeReserve); restore(out_brain->postbirth_constructor, 25u, offsets[25] == kNullArenaOffset ? 0u : 1u); restore(out_brain->retention_bank, 26u, offsets[26] == kNullArenaOffset ? 0u : out_brain->route_capacity);
    if (checkpoint.has_resource_ecology) {
      check_cuda(cudaMalloc(&out_brain->resource_ecology,
                            sizeof(checkpoint.resource_ecology)),
                 "restore ecology allocation");
      check_cuda(cudaMemcpy(out_brain->resource_ecology,
                            &checkpoint.resource_ecology,
                            sizeof(checkpoint.resource_ecology),
                            cudaMemcpyHostToDevice),
                 "restore resource ecology");
    }

    auto* restored_ecology = out_brain->resource_ecology;
    out_brain->resource_ecology = nullptr;
    DirectAdultRuntime* runtime = nullptr;
    try {
      runtime = create_direct_adult_runtime(out_brain, checkpoint.config);
      out_brain->resource_ecology = restored_ecology;
      const auto buffers = checkpoint_buffer_views(*runtime);
      for (std::size_t index = 0u; index < buffers.size(); ++index) {
        if (checkpoint.device_buffers[index].size() != buffers[index].bytes) {
          throw std::invalid_argument(
              "restore_direct_adult_checkpoint: runtime buffer size mismatch");
        }
        if (buffers[index].bytes != 0u) {
          check_cuda(cudaMemcpy(buffers[index].pointer,
                                checkpoint.device_buffers[index].data(),
                                buffers[index].bytes, cudaMemcpyHostToDevice),
                     "restore runtime buffer");
        }
      }
      install_restored_transport_indexes(*runtime, rebuilt_transport);

      std::copy(checkpoint.host_ingress_staging.begin(),
                checkpoint.host_ingress_staging.end(), runtime->host_ingress_staging);
      std::copy(checkpoint.host_ingress_contact_staging.begin(),
                checkpoint.host_ingress_contact_staging.end(),
                runtime->host_ingress_contact_staging);
      std::copy(checkpoint.host_consequence_staging.begin(),
                checkpoint.host_consequence_staging.end(),
                runtime->host_consequence_staging);
      runtime->resident_development_epochs = checkpoint.resident_development_epochs;
      runtime->current_tick = checkpoint.current_tick;
      runtime->host_ingress_write_tail = checkpoint.host_ingress_write_tail;
      runtime->host_ingress_publish_tail = checkpoint.host_ingress_publish_tail;
      runtime->host_ingress_observed_head = checkpoint.host_ingress_observed_head;
      runtime->host_ingress_dispatched_tail = checkpoint.host_ingress_dispatched_tail;
      runtime->host_ingress_overflow_drops = checkpoint.host_ingress_overflow_drops;
      runtime->host_ingress_protocol_faults = checkpoint.host_ingress_protocol_faults;
      runtime->host_consequence_write_tail = checkpoint.host_consequence_write_tail;
      runtime->host_consequence_publish_tail = checkpoint.host_consequence_publish_tail;
      runtime->host_consequence_observed_head = checkpoint.host_consequence_observed_head;
      runtime->host_consequence_overflow_drops =
          checkpoint.host_consequence_overflow_drops;
      runtime->host_consequence_protocol_faults =
          checkpoint.host_consequence_protocol_faults;
      *runtime->host_ingress_head_snapshot = checkpoint.host_ingress_head_snapshot;
      *runtime->host_ingress_publish_slot_pinned = checkpoint.host_ingress_publish_slot;
      *runtime->host_consequence_head_snapshot =
          checkpoint.host_consequence_head_snapshot;
      *runtime->host_consequence_publish_slot_pinned =
          checkpoint.host_consequence_publish_slot;
      runtime->execution_authority = AdultExecutionAuthority::host_stepped;
      runtime->is_persistent_running = false;
      return runtime;
    } catch (...) {
      out_brain->resource_ecology = restored_ecology;
      if (runtime != nullptr) destroy_direct_adult_runtime(runtime);
      throw;
    }
  } catch (...) {
    cudaFree(out_brain->resource_ecology);
    cudaFree(out_brain->arena);
    *out_brain = DirectBrain{};
    throw;
  }
}
