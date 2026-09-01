#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_SCHEDULE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_SCHEDULE_CUH
// Mid-include splice via direct_adult_device_ops.cuh.
// Owns delayed-packet identity/free-list, retire, and schedule.

__device__ inline bool inherit_eligibility_lineage(
    DirectParticipationDescriptor* child, std::uint32_t source_expiry_tick,
    const EligibilityRecord* records, const std::uint32_t* generations,
    std::uint32_t current_tick) {
  if (child == nullptr || source_expiry_tick < current_tick) return false;
  if (child->parent_eligibility_ref == 0u) {
    child->lineage_expiry_tick = source_expiry_tick;
    child->ancestry_depth = 1u;
    return true;
  }
  if (records == nullptr || generations == nullptr) return false;
  const std::uint32_t encoded_slot =
      static_cast<std::uint32_t>(child->parent_eligibility_ref);
  const std::uint32_t expected_generation =
      static_cast<std::uint32_t>(child->parent_eligibility_ref >> 32);
  if (encoded_slot == 0u || expected_generation == 0u) return false;
  const std::uint32_t slot = encoded_slot - 1u;
  if (slot >= kMaxLiveEligibilityRecords ||
      atomicAdd(const_cast<std::uint32_t*>(generations + slot), 0u) !=
          expected_generation)
    return false;
  const EligibilityRecord parent = records[slot];
  __threadfence();
  if (atomicAdd(const_cast<std::uint32_t*>(generations + slot), 0u) !=
          expected_generation ||
      parent.live == 0u || parent.expiry_tick < current_tick ||
      parent.lineage_expiry_tick < current_tick || parent.ancestry_depth == 0u ||
      parent.ancestry_depth >= kMaxProvenanceSlotsPerNode ||
      parent.ticket_id != child->ticket_id ||
      parent.target_node != child->source_node ||
      parent.claim_incarnation != child->claim_incarnation ||
      parent.authority != child->authority ||
      parent.authority_incarnation != child->authority_incarnation)
    return false;
  child->lineage_expiry_tick = parent.lineage_expiry_tick;
  child->ancestry_depth = parent.ancestry_depth + 1u;
  return true;
}

__device__ inline void update_delayed_packet_peak(
    std::uint32_t live, AdultCoreMetrics* metrics) {
  if (metrics == nullptr) return;
  unsigned long long* peak = reinterpret_cast<unsigned long long*>(
      &metrics->delayed_packets_pending_peak);
  unsigned long long observed = atomicAdd(peak, 0ULL);
  while (observed < live) {
    const unsigned long long prior = atomicCAS(peak, observed, live);
    if (prior == observed) return;
    observed = prior;
  }
}

DIRECT_ADULT_HD inline std::uint32_t delayed_packet_identity_hash(
    std::uint32_t route_index, std::uint32_t due_tick,
    std::uint64_t route_incarnation) {
  std::uint64_t value = route_incarnation ^
      (static_cast<std::uint64_t>(route_index) << 32) ^ due_tick;
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  return static_cast<std::uint32_t>(value) &
         (kDelayedPacketIdentityCapacity - 1u);
}

DIRECT_ADULT_HD inline bool delayed_packet_identity_matches(
    const DelayedPacketIdentity& identity, std::uint32_t route_index,
    std::uint32_t due_tick, std::uint64_t route_incarnation) {
  return identity.route_index == route_index && identity.due_tick == due_tick &&
         identity.route_incarnation == route_incarnation;
}

DIRECT_ADULT_HD inline bool delayed_packet_payload_matches(
    const DelayedSparsePacket& left, const DelayedSparsePacket& right) {
  if (left.due_tick != right.due_tick ||
      left.source_node != right.source_node ||
      left.target_node != right.target_node ||
      left.route_index != right.route_index ||
      left.context_signature != right.context_signature ||
      left.source_activation_q16 != right.source_activation_q16 ||
      left.drive_q16 != right.drive_q16 ||
      left.reserved_arrival_eligibility_q16 !=
          right.reserved_arrival_eligibility_q16 ||
      left.source_ancestry_incomplete != right.source_ancestry_incomplete ||
      left.claim_count != right.claim_count ||
      left.route_incarnation != right.route_incarnation)
    return false;
  for (std::uint32_t i = 0u; i < kMaxProvenanceSlotsPerNode; ++i)
    if (left.source_claims[i].ticket_id != right.source_claims[i].ticket_id ||
        left.source_claims[i].source_node !=
            right.source_claims[i].source_node ||
        left.source_claims[i].target_node !=
            right.source_claims[i].target_node ||
        left.source_claims[i].route_index !=
            right.source_claims[i].route_index ||
        left.source_claims[i].context_signature !=
            right.source_claims[i].context_signature ||
        left.source_claims[i].expiry_tick !=
            right.source_claims[i].expiry_tick ||
        left.source_claims[i].claim_incarnation !=
            right.source_claims[i].claim_incarnation ||
        left.source_claims[i].route_incarnation !=
            right.source_claims[i].route_incarnation ||
        left.source_claims[i].authority != right.source_claims[i].authority ||
        left.source_claims[i].authority_incarnation !=
            right.source_claims[i].authority_incarnation ||
        left.source_claims[i].contribution_kind !=
            right.source_claims[i].contribution_kind ||
        left.source_claims[i].eligibility_slot !=
            right.source_claims[i].eligibility_slot ||
        left.source_claims[i].eligibility_generation !=
            right.source_claims[i].eligibility_generation ||
        left.source_claims[i].frozen_eligibility_q16 !=
            right.source_claims[i].frozen_eligibility_q16 ||
        left.source_claims[i].parent_eligibility_ref !=
            right.source_claims[i].parent_eligibility_ref ||
        left.source_claims[i].lineage_expiry_tick !=
            right.source_claims[i].lineage_expiry_tick ||
        left.source_claims[i].ancestry_depth !=
            right.source_claims[i].ancestry_depth)
      return false;
  return true;
}

DIRECT_ADULT_HD inline bool delayed_packet_causal_payload_structural(
    const DelayedSparsePacket& packet, std::uint32_t node_count,
    std::uint32_t route_capacity) {
  if (packet.due_tick == 0u || packet.source_node >= node_count ||
      packet.target_node >= node_count ||
      packet.route_index >= route_capacity ||
      packet.route_incarnation == 0u ||
      (packet.drive_q16 != 0 && packet.claim_count == 0u &&
       packet.source_ancestry_incomplete == 0u) ||
      packet.claim_count > kMaxProvenanceSlotsPerNode)
    return false;
  for (std::uint32_t i = 0u; i < packet.claim_count; ++i) {
    const DirectParticipationDescriptor& claim = packet.source_claims[i];
    if (claim.ticket_id == 0u || claim.ticket_id == kInvalidTicket ||
        claim.source_node != packet.source_node ||
        claim.target_node != packet.target_node ||
        claim.route_index != packet.route_index ||
        claim.context_signature != packet.context_signature ||
        claim.route_incarnation != packet.route_incarnation ||
        claim.authority == DirectParticipationAuthority::none ||
        claim.contribution_kind != DirectContributionKind::sparse_route ||
        claim.ancestry_depth == 0u ||
        claim.ancestry_depth > kMaxProvenanceSlotsPerNode)
      return false;
  }
  return true;
}

DIRECT_ADULT_HD inline bool delayed_packet_causal_payload_current(
    const DelayedSparsePacket& packet, std::uint32_t current_tick,
    std::uint32_t node_count, std::uint32_t route_capacity) {
  if (packet.due_tick <= current_tick ||
      packet.due_tick - current_tick > kMaxPhysicalRouteDelayTicks ||
      !delayed_packet_causal_payload_structural(
          packet, node_count, route_capacity))
    return false;
  for (std::uint32_t i = 0u; i < packet.claim_count; ++i)
    if (packet.source_claims[i].expiry_tick < current_tick ||
        packet.source_claims[i].lineage_expiry_tick < current_tick)
      return false;
  return true;
}

DIRECT_ADULT_HD inline bool delayed_packet_publishable(
    const DelayedSparsePacket& packet, std::uint32_t current_tick,
    std::uint32_t node_count, std::uint32_t route_capacity) {
  return packet.live ==
             static_cast<std::uint32_t>(DelayedPacketSlotState::reserved) &&
         delayed_packet_causal_payload_current(
             packet, current_tick, node_count, route_capacity);
}

__device__ inline void release_delayed_packet_identity(
    DelayedPacketIdentity* identities, std::uint32_t identity_slot,
    std::uint32_t route_index, std::uint32_t due_tick,
    std::uint64_t route_incarnation) {
  if (identities == nullptr || identity_slot >= kDelayedPacketIdentityCapacity)
    return;
  DelayedPacketIdentity& entry = identities[identity_slot];
  if (!delayed_packet_identity_matches(entry, route_index, due_tick,
                                       route_incarnation))
    return;
  entry = DelayedPacketIdentity{};
}

__device__ inline void retire_delayed_packet(
    DelayedSparsePacket* packets, DelayedPacketIdentity* identities,
    std::uint32_t packet_index) {
  const DelayedSparsePacket packet = packets[packet_index];
  release_delayed_packet_identity(
      identities, packet.reserved0, packet.route_index, packet.due_tick,
      packet.route_incarnation);
  packets[packet_index].live =
      static_cast<std::uint32_t>(DelayedPacketSlotState::free);
}

__device__ inline void propose_delayed_sparse_packet(
    RouteTransportProposal* proposals, std::uint32_t route_capacity,
    const NodeCausalParticipation* source_frontier,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations,
    std::uint32_t* target_ancestry_incomplete,
    std::uint32_t source_node, std::uint32_t target_node,
    std::uint32_t route_index, std::uint64_t route_incarnation,
    std::uint32_t context_signature, std::int32_t source_activation_q16,
    std::int32_t drive_q16, std::uint32_t source_ancestry_incomplete,
    std::uint32_t due_tick, std::uint32_t current_tick,
    const std::uint32_t* live_spans = nullptr,
    std::uint32_t span_node_count = 0u) {
  if (proposals == nullptr || source_frontier == nullptr ||
      route_index >= route_capacity || due_tick <= current_tick ||
      due_tick - current_tick > kMaxPhysicalRouteDelayTicks)
    return;
  RouteTransportProposal proposal{};
  proposal.packet.live =
      static_cast<std::uint32_t>(DelayedPacketSlotState::reserved);
  proposal.packet.due_tick = due_tick;
  proposal.packet.source_node = source_node;
  proposal.packet.target_node = target_node;
  proposal.packet.route_index = route_index;
  proposal.packet.context_signature = context_signature;
  proposal.packet.source_activation_q16 = source_activation_q16;
  proposal.packet.drive_q16 = drive_q16;
  proposal.packet.source_ancestry_incomplete = source_ancestry_incomplete;
  proposal.packet.route_incarnation = route_incarnation;
  bool all_claims_lawful = true;
  std::uint32_t limit = kNodeParticipationAperture;
  if (live_spans != nullptr && source_node < span_node_count) {
    const std::uint32_t span = live_spans[source_node];
    limit = span == 0u
                ? 0u
                : (span < kNodeParticipationAperture ? span
                                                     : kNodeParticipationAperture);
  }
  for (std::uint32_t s = 0u; s < limit; ++s) {
    const NodeCausalParticipation claim = read_participation_slot(
        source_frontier + source_node * kNodeParticipationAperture + s);
    if (claim.current_drive == 0u || claim.expiry_tick < current_tick)
      continue;
    if (claim.ticket_id == 0ULL || claim.ticket_id == kInvalidTicket ||
        claim.authority == DirectParticipationAuthority::none) {
      all_claims_lawful = false;
      continue;
    }
    if (proposal.packet.claim_count >= kRouteTransportProducerWidth) {
      all_claims_lawful = false;
      break;
    }
    DirectParticipationDescriptor frozen{};
    frozen.ticket_id = claim.ticket_id;
    frozen.source_node = source_node;
    frozen.target_node = target_node;
    frozen.route_index = route_index;
    frozen.context_signature = context_signature;
    frozen.expiry_tick = claim.expiry_tick;
    frozen.claim_incarnation = claim.claim_incarnation;
    frozen.route_incarnation = route_incarnation;
    frozen.authority = claim.authority;
    frozen.authority_incarnation = claim.authority_incarnation;
    frozen.contribution_kind = DirectContributionKind::sparse_route;
    frozen.parent_eligibility_ref = participation_eligibility_tail(claim);
    if (!inherit_eligibility_lineage(
            &frozen, claim.expiry_tick, eligibility_table,
            eligibility_record_generations, current_tick)) {
      all_claims_lawful = false;
      continue;
    }
    proposal.packet.source_claims[proposal.packet.claim_count++] = frozen;
  }
  const bool causal_incomplete = drive_q16 != 0 &&
      (source_ancestry_incomplete != 0u || !all_claims_lawful ||
       proposal.packet.claim_count == 0u);
  proposal.packet.source_ancestry_incomplete = causal_incomplete ? 1u : 0u;
  if (causal_incomplete &&
      target_ancestry_incomplete != nullptr)
    atomicExch(target_ancestry_incomplete, 1u);
  proposal.valid = 1u;
  proposals[route_index] = proposal;
}

template <typename Grid>
__device__ inline std::uint32_t route_transport_exclusive_scan(
    Grid grid, std::uint32_t* values, std::uint32_t* scratch,
    std::uint32_t count) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  std::uint32_t* source = values;
  std::uint32_t* target = scratch;
  for (std::uint32_t offset = 1u; offset < count; offset <<= 1u) {
    for (std::uint32_t i = tid; i < count; i += stride)
      target[i] = source[i] + (i >= offset ? source[i - offset] : 0u);
    grid.sync();
    std::uint32_t* swap = source;
    source = target;
    target = swap;
  }
  const std::uint32_t total = count == 0u ? 0u : source[count - 1u];
  for (std::uint32_t i = tid; i < count; i += stride)
    target[i] = i == 0u ? 0u : source[i - 1u];
  grid.sync();
  if (target != values) {
    for (std::uint32_t i = tid; i < count; i += stride) values[i] = target[i];
    grid.sync();
  }
  return total;
}

__device__ inline std::uint32_t route_transport_sort_extent(
    std::uint32_t count) {
  std::uint32_t extent = 1u;
  while (extent < count) extent <<= 1u;
  return extent;
}

template <typename Grid, typename Less>
__device__ inline void route_transport_bitonic_sort_indices(
    Grid grid, std::uint32_t* indices, std::uint32_t count,
    std::uint32_t capacity, Less less) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  const std::uint32_t extent = route_transport_sort_extent(count);
  if (extent > capacity) return;
  for (std::uint32_t i = tid + count; i < extent; i += stride)
    indices[i] = kInvalidIndex;
  grid.sync();
  for (std::uint32_t width = 2u; width <= extent; width <<= 1u) {
    for (std::uint32_t distance = width >> 1u; distance != 0u;
         distance >>= 1u) {
      for (std::uint32_t i = tid; i < extent; i += stride) {
        const std::uint32_t peer = i ^ distance;
        if (peer <= i) continue;
        const std::uint32_t a = indices[i];
        const std::uint32_t b = indices[peer];
        const bool ascending = (i & width) == 0u;
        const bool a_less_b = less(a, b);
        const bool b_less_a = less(b, a);
        if ((ascending && b_less_a) || (!ascending && a_less_b)) {
          indices[i] = b;
          indices[peer] = a;
        }
      }
      grid.sync();
    }
  }
}

struct DelayedPacketChronologyLess {
  const DelayedSparsePacket* packets;
  __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
    if (a == kInvalidIndex) return false;
    if (b == kInvalidIndex) return true;
    const DelayedSparsePacket& left = packets[a];
    const DelayedSparsePacket& right = packets[b];
    if (left.route_index != right.route_index)
      return left.route_index < right.route_index;
    if (left.due_tick != right.due_tick)
      return left.due_tick < right.due_tick;
    if (left.route_incarnation != right.route_incarnation)
      return left.route_incarnation < right.route_incarnation;
    return a < b;
  }
};

struct RouteProposalChronologyLess {
  const RouteTransportProposal* proposals;
  std::uint32_t rotation;
  std::uint32_t route_capacity;
  __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
    if (a == kInvalidIndex) return false;
    if (b == kInvalidIndex) return true;
    const RouteTransportProposal& left = proposals[a];
    const RouteTransportProposal& right = proposals[b];
    if (left.packet.due_tick != right.packet.due_tick)
      return left.packet.due_tick < right.packet.due_tick;
    const std::uint32_t left_rank =
        (a + route_capacity - rotation) % route_capacity;
    const std::uint32_t right_rank =
        (b + route_capacity - rotation) % route_capacity;
    return left_rank != right_rank ? left_rank < right_rank : a < b;
  }
};

enum class DelayedPacketLookup : std::uint32_t {
  absent = 0u,
  exact_duplicate = 1u,
  payload_mismatch = 2u,
  unresolved_index = 3u,
};

__device__ inline DelayedPacketLookup route_transport_packet_lookup(
    const DelayedPacketIdentity* identities,
    const DelayedSparsePacket* packets,
    const DelayedSparsePacket& packet) {
  if (identities == nullptr || packets == nullptr)
    return DelayedPacketLookup::unresolved_index;
  const std::uint32_t start = delayed_packet_identity_hash(
      packet.route_index, packet.due_tick, packet.route_incarnation);
  bool saw_free = false;
  for (std::uint32_t probe = 0u; probe < kDelayedPacketIdentityMaxProbes;
       ++probe) {
    const DelayedPacketIdentity identity =
        identities[(start + probe) & (kDelayedPacketIdentityCapacity - 1u)];
    if (identity.state ==
            static_cast<std::uint32_t>(DelayedPacketIdentityState::claiming))
      return DelayedPacketLookup::unresolved_index;
    if (identity.state ==
        static_cast<std::uint32_t>(DelayedPacketIdentityState::free)) {
      if (identity.packet_index_plus_one != 0u)
        return DelayedPacketLookup::unresolved_index;
      saw_free = true;
      continue;
    }
    if (identity.state !=
        static_cast<std::uint32_t>(DelayedPacketIdentityState::published))
      return DelayedPacketLookup::unresolved_index;
    if (!delayed_packet_identity_matches(
            identity, packet.route_index, packet.due_tick,
            packet.route_incarnation))
      continue;
    if (identity.packet_index_plus_one == 0u ||
        identity.packet_index_plus_one > kMaxDelayedSparsePackets)
      return DelayedPacketLookup::unresolved_index;
    const DelayedSparsePacket& prior =
        packets[identity.packet_index_plus_one - 1u];
    if (prior.live !=
            static_cast<std::uint32_t>(DelayedPacketSlotState::published) ||
        prior.route_index != packet.route_index ||
        prior.due_tick != packet.due_tick ||
        prior.route_incarnation != packet.route_incarnation)
      return DelayedPacketLookup::unresolved_index;
    return delayed_packet_payload_matches(prior, packet)
               ? DelayedPacketLookup::exact_duplicate
               : DelayedPacketLookup::payload_mismatch;
  }
  return saw_free ? DelayedPacketLookup::absent
                  : DelayedPacketLookup::unresolved_index;
}

template <typename Grid>
__device__ inline void compact_delayed_packet_frontier(
    Grid grid, DelayedSparsePacket* packets,
    std::uint32_t active_count, std::uint32_t* live_count,
    std::uint32_t* free_head, std::uint32_t* frontier,
    std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t* rebuilt_frontier) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  for (std::uint32_t rank = tid; rank < active_count; rank += stride) {
    const std::uint32_t slot = frontier[rank];
    scan[rank] =
        slot < kMaxDelayedSparsePackets &&
                packets[slot].live ==
                    static_cast<std::uint32_t>(
                        DelayedPacketSlotState::published)
            ? 1u
            : 0u;
  }
  grid.sync();
  const std::uint32_t survivors = route_transport_exclusive_scan(
      grid, scan, scratch, active_count);
  const std::uint32_t old_free = kMaxDelayedSparsePackets - active_count;
  for (std::uint32_t rank = tid; rank < active_count; rank += stride) {
    const std::uint32_t slot = frontier[rank];
    if (scan[rank] < survivors &&
        slot < kMaxDelayedSparsePackets &&
        packets[slot].live ==
            static_cast<std::uint32_t>(
                DelayedPacketSlotState::published)) {
      rebuilt_frontier[scan[rank]] = slot;
    } else {
      rebuilt_frontier[
          survivors + old_free + rank - scan[rank]] = slot;
    }
  }
  for (std::uint32_t rank = active_count + tid;
       rank < kMaxDelayedSparsePackets; rank += stride)
    rebuilt_frontier[survivors + rank - active_count] = frontier[rank];
  grid.sync();
  for (std::uint32_t rank = tid; rank < kMaxDelayedSparsePackets;
       rank += stride)
    frontier[rank] = rebuilt_frontier[rank];
  grid.sync();
  if (tid == 0u) {
    *live_count = survivors;
    *free_head = survivors < kMaxDelayedSparsePackets
                     ? frontier[survivors]
                     : kInvalidIndex;
  }
  grid.sync();
}

template <typename Grid>
__device__ inline void materialize_route_transport_proposals(
    Grid grid, RouteTransportProposal* proposals,
    std::uint32_t route_capacity, std::uint32_t current_tick,
    std::uint32_t node_count,
    std::uint32_t* cursor, DelayedSparsePacket* packets,
    std::uint32_t* live_count, std::uint32_t* free_head,
    std::uint32_t* next_free, DelayedPacketIdentity* identities,
    std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t scratch_capacity, std::uint32_t* free_slots,
    std::uint32_t* identity_owners,
    std::uint32_t* node_next_ancestry_incomplete,
    AdultCoreMetrics* metrics) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  if (proposals == nullptr || packets == nullptr || live_count == nullptr ||
      free_head == nullptr || next_free == nullptr || identities == nullptr ||
      scan == nullptr || scratch == nullptr || free_slots == nullptr ||
      identity_owners == nullptr)
    return;
  const std::uint32_t old_live = *live_count;
  if (old_live > kMaxDelayedSparsePackets) return;
  const std::uint32_t free_count = kMaxDelayedSparsePackets - old_live;

  const std::uint32_t rotation =
      cursor != nullptr && route_capacity != 0u ? *cursor % route_capacity : 0u;
  for (std::uint32_t route = tid; route < route_capacity; route += stride) {
    const RouteTransportProposal proposal = proposals[route];
    const bool canonical =
        proposal.valid != 0u && proposal.packet.route_index == route &&
        delayed_packet_publishable(
            proposal.packet, current_tick, node_count, route_capacity);
    if (!canonical) {
      proposals[route].valid = 0u;
      if (proposal.valid != 0u) {
        if (metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->delayed_packets_overflow_rejections),
                    1ULL);
        if (proposal.packet.drive_q16 != 0 &&
            proposal.packet.target_node < node_count &&
            node_next_ancestry_incomplete != nullptr)
          atomicExch(
              node_next_ancestry_incomplete + proposal.packet.target_node,
              1u);
      }
    }
    const DelayedPacketLookup lookup =
        canonical
            ? route_transport_packet_lookup(
                  identities, packets, proposal.packet)
            : DelayedPacketLookup::absent;
    scan[route] = canonical &&
                          lookup == DelayedPacketLookup::absent
                      ? 1u
                      : 0u;
    if (canonical && metrics != nullptr) {
      if (lookup == DelayedPacketLookup::exact_duplicate ||
          lookup == DelayedPacketLookup::payload_mismatch)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->delayed_packets_duplicate_rejections),
                  1ULL);
      else if (lookup == DelayedPacketLookup::unresolved_index)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->delayed_packets_overflow_rejections),
                  1ULL);
    }
    if ((lookup == DelayedPacketLookup::payload_mismatch ||
         lookup == DelayedPacketLookup::unresolved_index) &&
        proposal.packet.drive_q16 != 0 &&
        (proposal.packet.source_ancestry_incomplete != 0u ||
         proposal.packet.claim_count != 0u) &&
        node_next_ancestry_incomplete != nullptr)
      atomicExch(node_next_ancestry_incomplete +
                     proposal.packet.target_node,
                 1u);
  }
  grid.sync();
  const std::uint32_t candidate_count = route_transport_exclusive_scan(
      grid, scan, scratch, route_capacity);
  for (std::uint32_t route = tid; route < route_capacity; route += stride)
    if (scan[route] < candidate_count &&
        proposals[route].valid != 0u &&
        route_transport_packet_lookup(identities, packets,
                                      proposals[route].packet) ==
            DelayedPacketLookup::absent)
      scratch[scan[route]] = route;
  grid.sync();
  route_transport_bitonic_sort_indices(
      grid, scratch, candidate_count, scratch_capacity,
      RouteProposalChronologyLess{proposals, rotation, route_capacity});
  const std::uint32_t reserved =
      candidate_count < free_count ? candidate_count : free_count;
  if (tid == 0u && cursor != nullptr && route_capacity != 0u) {
    const std::uint32_t pivot =
        candidate_count == 0u
            ? rotation
            : scratch[reserved == 0u ? 0u : reserved - 1u];
    *cursor = (pivot + 1u) % route_capacity;
  }
  grid.sync();

  for (std::uint32_t rank = tid; rank < candidate_count; rank += stride) {
    const std::uint32_t route = scratch[rank];
    const RouteTransportProposal proposal = proposals[route];
    if (rank < reserved) {
      const std::uint32_t slot = next_free[old_live + rank];
      DelayedSparsePacket packet = proposal.packet;
      packet.live =
          static_cast<std::uint32_t>(DelayedPacketSlotState::reserved);
      packet.reserved0 = kInvalidIndex;
      packets[slot] = packet;
    } else {
      const bool causal_pressure =
          proposal.packet.drive_q16 != 0 &&
          (proposal.packet.source_ancestry_incomplete != 0u ||
           proposal.packet.claim_count != 0u);
      if (causal_pressure && node_next_ancestry_incomplete != nullptr)
        atomicExch(node_next_ancestry_incomplete +
                       proposal.packet.target_node,
                   1u);
    }
  }
  // Every reserved packet must be frozen before any thread can clear a route
  // proposal. Without this boundary, the copy and clear loops tear the same
  // proposal across threads after chronological sorting.
  grid.sync();
  for (std::uint32_t route = tid; route < route_capacity; route += stride)
    proposals[route] = RouteTransportProposal{};
  grid.sync();

  constexpr std::uint32_t kDuplicateResolution = kInvalidIndex - 1u;
  for (std::uint32_t rank = tid; rank < reserved; rank += stride) {
    scan[rank] = rank;
    free_slots[rank] = kInvalidIndex;
  }
  grid.sync();
  std::uint32_t unresolved = reserved;
  std::uint32_t* unresolved_now = scan;
  std::uint32_t* unresolved_next = scratch;
  for (std::uint32_t probe = 0u;
       probe < kDelayedPacketIdentityMaxProbes && unresolved != 0u;
       ++probe) {
    for (std::uint32_t i = tid; i < unresolved; i += stride) {
      const std::uint32_t rank = unresolved_now[i];
      const DelayedSparsePacket& packet =
          packets[next_free[old_live + rank]];
      const std::uint32_t identity_slot =
          (delayed_packet_identity_hash(
               packet.route_index, packet.due_tick,
               packet.route_incarnation) +
           probe) &
          (kDelayedPacketIdentityCapacity - 1u);
      DelayedPacketIdentity& identity = identities[identity_slot];
      if (identity.state ==
          static_cast<std::uint32_t>(DelayedPacketIdentityState::free))
        atomicCAS(&identity.packet_index_plus_one, 0u, ~0u);
    }
    grid.sync();
    for (std::uint32_t i = tid; i < unresolved; i += stride) {
      const std::uint32_t rank = unresolved_now[i];
      const DelayedSparsePacket& packet =
          packets[next_free[old_live + rank]];
      const std::uint32_t identity_slot =
          (delayed_packet_identity_hash(
               packet.route_index, packet.due_tick,
               packet.route_incarnation) +
           probe) &
          (kDelayedPacketIdentityCapacity - 1u);
      DelayedPacketIdentity& identity = identities[identity_slot];
      if (identity.state ==
          static_cast<std::uint32_t>(DelayedPacketIdentityState::free))
        atomicMin(&identity.packet_index_plus_one, rank + 1u);
    }
    grid.sync();
    if (tid == 0u) identity_owners[0] = 0u;
    grid.sync();
    for (std::uint32_t i = tid; i < unresolved; i += stride) {
      const std::uint32_t rank = unresolved_now[i];
      const std::uint32_t packet_slot = next_free[old_live + rank];
      const DelayedSparsePacket& packet = packets[packet_slot];
      const std::uint32_t identity_slot =
          (delayed_packet_identity_hash(
               packet.route_index, packet.due_tick,
               packet.route_incarnation) +
           probe) &
          (kDelayedPacketIdentityCapacity - 1u);
      const DelayedPacketIdentity identity = identities[identity_slot];
      if (identity.state ==
              static_cast<std::uint32_t>(
                  DelayedPacketIdentityState::free) &&
          identity.packet_index_plus_one == rank + 1u) {
        free_slots[rank] = identity_slot;
        continue;
      }

      bool duplicate = false;
      bool payload_mismatch = false;
      if (identity.state ==
              static_cast<std::uint32_t>(
                  DelayedPacketIdentityState::published) &&
          delayed_packet_identity_matches(
              identity, packet.route_index, packet.due_tick,
              packet.route_incarnation) &&
          identity.packet_index_plus_one != 0u &&
          identity.packet_index_plus_one <= kMaxDelayedSparsePackets) {
        duplicate = true;
        payload_mismatch = !delayed_packet_payload_matches(
            packets[identity.packet_index_plus_one - 1u], packet);
      } else if (
          identity.state ==
              static_cast<std::uint32_t>(
                  DelayedPacketIdentityState::free) &&
          identity.packet_index_plus_one != 0u &&
          identity.packet_index_plus_one != ~0u &&
          identity.packet_index_plus_one <= reserved) {
        const DelayedSparsePacket& owner =
            packets[next_free[
                old_live + identity.packet_index_plus_one - 1u]];
        if (owner.route_index == packet.route_index &&
            owner.due_tick == packet.due_tick &&
            owner.route_incarnation == packet.route_incarnation) {
          duplicate = true;
          payload_mismatch =
              !delayed_packet_payload_matches(owner, packet);
        }
      }
      if (duplicate) {
        free_slots[rank] = kDuplicateResolution;
        if (metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->delayed_packets_duplicate_rejections),
                    1ULL);
        if (payload_mismatch && packet.drive_q16 != 0 &&
            (packet.source_ancestry_incomplete != 0u ||
             packet.claim_count != 0u) &&
            node_next_ancestry_incomplete != nullptr)
          atomicExch(node_next_ancestry_incomplete + packet.target_node, 1u);
      } else {
        unresolved_next[atomicAdd(identity_owners, 1u)] = rank;
      }
    }
    grid.sync();
    for (std::uint32_t i = tid; i < unresolved; i += stride) {
      const std::uint32_t rank = unresolved_now[i];
      const std::uint32_t identity_slot = free_slots[rank];
      if (identity_slot >= kDelayedPacketIdentityCapacity) continue;
      const std::uint32_t packet_slot = next_free[old_live + rank];
      DelayedSparsePacket& packet = packets[packet_slot];
      packet.reserved0 = identity_slot;
      identities[identity_slot] = DelayedPacketIdentity{
          static_cast<std::uint32_t>(
              DelayedPacketIdentityState::published),
          packet_slot + 1u, packet.route_index, packet.due_tick,
          packet.route_incarnation};
    }
    grid.sync();
    unresolved = identity_owners[0];
    std::uint32_t* swap = unresolved_now;
    unresolved_now = unresolved_next;
    unresolved_next = swap;
  }

  for (std::uint32_t i = tid; i < unresolved; i += stride) {
    const DelayedSparsePacket& packet =
        packets[next_free[old_live + unresolved_now[i]]];
    if (packet.drive_q16 != 0 &&
        (packet.source_ancestry_incomplete != 0u ||
         packet.claim_count != 0u) &&
        node_next_ancestry_incomplete != nullptr)
      atomicExch(node_next_ancestry_incomplete + packet.target_node, 1u);
  }
  grid.sync();

  if (tid == 0u) identity_owners[0] = 0u;
  grid.sync();
  for (std::uint32_t rank = tid; rank < reserved; rank += stride) {
    const std::uint32_t identity_slot = free_slots[rank];
    if (identity_slot >= kDelayedPacketIdentityCapacity) continue;
    const std::uint32_t packet_slot = next_free[old_live + rank];
    bool canonical = packet_slot < kMaxDelayedSparsePackets;
    DelayedSparsePacket packet{};
    DelayedPacketIdentity identity{};
    if (canonical) {
      packet = packets[packet_slot];
      identity = identities[identity_slot];
      canonical =
          delayed_packet_publishable(
              packet, current_tick, node_count, route_capacity) &&
          packet.reserved0 == identity_slot &&
          identity.state == static_cast<std::uint32_t>(
                                DelayedPacketIdentityState::published) &&
          identity.packet_index_plus_one == packet_slot + 1u &&
          delayed_packet_identity_matches(
              identity, packet.route_index, packet.due_tick,
              packet.route_incarnation);
    }
    if (canonical) continue;
    if (packet_slot < kMaxDelayedSparsePackets) {
      if (identity.packet_index_plus_one == packet_slot + 1u)
        identities[identity_slot] = DelayedPacketIdentity{};
      packets[packet_slot].live =
          static_cast<std::uint32_t>(DelayedPacketSlotState::free);
      if (packet.drive_q16 != 0 && packet.target_node < node_count &&
          node_next_ancestry_incomplete != nullptr)
        atomicExch(
            node_next_ancestry_incomplete + packet.target_node, 1u);
    }
    free_slots[rank] = kInvalidIndex;
    atomicAdd(identity_owners, 1u);
  }
  grid.sync();
  const std::uint32_t final_invalid = identity_owners[0];

  for (std::uint32_t rank = tid; rank < reserved; rank += stride)
    scan[rank] =
        free_slots[rank] < kDelayedPacketIdentityCapacity ? 1u : 0u;
  grid.sync();
  const std::uint32_t materialized = route_transport_exclusive_scan(
      grid, scan, scratch, reserved);
  for (std::uint32_t rank = tid; rank < reserved; rank += stride) {
    const std::uint32_t packet_slot = next_free[old_live + rank];
    if (scan[rank] < materialized &&
        free_slots[rank] < kDelayedPacketIdentityCapacity) {
      packets[packet_slot].live =
          static_cast<std::uint32_t>(
              DelayedPacketSlotState::published);
    } else {
      packets[packet_slot].live =
          static_cast<std::uint32_t>(DelayedPacketSlotState::free);
    }
  }
  grid.sync();

  // The allocator frontier is a derived index, not checkpoint authority.
  // Canonicalize it from the published packet slots at every phase boundary so
  // restore can rebuild the identical bytes without trusting saved scratch.
  static_assert(kRouteTransportOwnerScratchCapacity >=
                kMaxDelayedSparsePackets);
  for (std::uint32_t slot = tid; slot < kMaxDelayedSparsePackets;
       slot += stride)
    scan[slot] =
        packets[slot].live == static_cast<std::uint32_t>(
                                  DelayedPacketSlotState::published)
            ? 1u
            : 0u;
  grid.sync();
  const std::uint32_t canonical_live = route_transport_exclusive_scan(
      grid, scan, scratch, kMaxDelayedSparsePackets);
  for (std::uint32_t slot = tid; slot < kMaxDelayedSparsePackets;
       slot += stride) {
    const std::uint32_t published_before = scan[slot];
    const bool published =
        packets[slot].live == static_cast<std::uint32_t>(
                                  DelayedPacketSlotState::published);
    const std::uint32_t rank =
        published ? published_before
                  : canonical_live + slot - published_before;
    identity_owners[rank] = slot;
  }
  grid.sync();
  for (std::uint32_t rank = tid; rank < kMaxDelayedSparsePackets;
       rank += stride)
    next_free[rank] = identity_owners[rank];
  grid.sync();
  if (tid == 0u) {
    *live_count = canonical_live;
    *free_head = canonical_live < kMaxDelayedSparsePackets
                     ? next_free[canonical_live]
                     : kInvalidIndex;
    if (metrics != nullptr) {
      metrics->delayed_packets_overflow_rejections +=
          candidate_count - reserved + unresolved + final_invalid;
      metrics->delayed_packets_scheduled += materialized;
      update_delayed_packet_peak(canonical_live, metrics);
    }
  }
  grid.sync();
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_SCHEDULE_CUH
