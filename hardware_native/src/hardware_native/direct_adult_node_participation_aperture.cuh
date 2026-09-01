#ifndef HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_APERTURE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_APERTURE_CUH
// #1610 N=512 node-participation residency aperture: live-span, match, meters,
// free-slot admit, publish, stage. Owned here so delayed_sparse_delivery stays
// a thin composition point.

// Live high-water (max occupied slot index + 1). Empty past span is never
// written while lowest-free admit holds; holes inside span still need scans.
template <typename Grid>
__device__ inline void compute_node_participation_live_spans(
    Grid grid, const NodeCausalParticipation* participation,
    std::uint32_t* spans, std::uint32_t node_count,
    std::uint32_t current_tick) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  if (participation == nullptr || spans == nullptr || node_count == 0u) return;
  for (std::uint32_t node = tid; node < node_count; node += stride)
    spans[node] = 0u;
  grid.sync();
  const std::uint32_t node_slots = node_count * kNodeParticipationAperture;
  for (std::uint32_t i = tid; i < node_slots; i += stride) {
    const std::uint32_t node = i / kNodeParticipationAperture;
    const std::uint32_t slot = i - node * kNodeParticipationAperture;
    const NodeCausalParticipation current =
        read_participation_slot(participation + i);
    if (current.ticket_id != 0ull && current.expiry_tick >= current_tick)
      atomicMax(spans + node, slot + 1u);
  }
  grid.sync();
}

__device__ inline std::uint32_t node_participation_span_limit(
    const std::uint32_t* spans, std::uint32_t node, std::uint32_t node_count) {
  if (spans == nullptr || node >= node_count) return kNodeParticipationAperture;
  const std::uint32_t span = spans[node];
  return span == 0u ? 0u
                    : (span < kNodeParticipationAperture
                           ? span
                           : kNodeParticipationAperture);
}

template <typename Grid>
__device__ inline void admit_node_participation_aperture(
    Grid grid, EligibilityBatchClaim* claims, std::uint32_t claim_count,
    EligibilityRecord* records, std::uint32_t* scan, std::uint32_t* scratch,
    NodeCausalParticipation* node_next_participation,
    std::uint32_t* node_next_ancestry_incomplete,
    std::uint32_t* node_candidate_owners, std::uint32_t node_count,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity, std::uint32_t current_tick,
    AdultCoreMetrics* metrics) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  if (claims == nullptr || records == nullptr || scan == nullptr ||
      scratch == nullptr || node_candidate_owners == nullptr ||
      node_next_participation == nullptr)
    return;

  // spans[node] in owners[0..node_count); slot races use owners[node_count..).
  const std::uint32_t node_slots = node_count * kNodeParticipationAperture;
  std::uint32_t* spans = node_candidate_owners;
  compute_node_participation_live_spans(grid, node_next_participation, spans,
                                        node_count, current_tick);
  for (std::uint32_t i = tid; i < node_slots; i += stride) {
    if (i >= node_count) node_candidate_owners[i] = kInvalidIndex;
  }
  grid.sync();

  // Exact refresh / collision detect over live span only.
  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    claims[i].admitted = 0u;
    if (claims[i].valid == 0u || claims[i].canonical_producer != i ||
        claims[i].eligibility_slot == kInvalidIndex ||
        claims[i].descriptor.target_node >= node_count)
      continue;
    const std::uint32_t node = claims[i].descriptor.target_node;
    const std::uint32_t limit = node_participation_span_limit(spans, node,
                                                             node_count);
    if (limit == 0u) {
      claims[i].admitted = 0u;
      continue;
    }
    const std::uint64_t tail = eligibility_record_ref(
        claims[i].eligibility_slot, claims[i].eligibility_generation);
    const std::uint32_t base = node * kNodeParticipationAperture;
    std::uint32_t matches = 0u;
    for (std::uint32_t slot = 0u; slot < limit; ++slot) {
      const NodeCausalParticipation current =
          read_participation_slot(node_next_participation + base + slot);
      const bool match = participation_slot_exact_match(
          current, claims[i].descriptor, tail, current_tick);
      if (!match) continue;
      claims[i].admitted = slot + 1u;
      ++matches;
      if (matches >= 2u) break;
    }
    if (matches > 1u) {
      claims[i].admitted = kInvalidIndex;
      if (node_next_ancestry_incomplete != nullptr)
        atomicExch(node_next_ancestry_incomplete + node, 1u);
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->participation_authority_collision_rejects),
                  1ULL);
    } else if (matches == 0u) {
      claims[i].admitted = 0u;
    }
  }
  grid.sync();

  // Observer uncapped demand (independent of N). Unmatched in owners[node+N].
  if (metrics != nullptr) {
    for (std::uint32_t node = tid; node < node_count; node += stride)
      node_candidate_owners[node_count + node] = 0u;
    grid.sync();
    for (std::uint32_t i = tid; i < claim_count; i += stride) {
      if (claims[i].valid == 0u || claims[i].canonical_producer != i ||
          claims[i].eligibility_slot == kInvalidIndex ||
          claims[i].admitted != 0u ||
          claims[i].descriptor.target_node >= node_count)
        continue;
      atomicAdd(node_candidate_owners + node_count +
                    claims[i].descriptor.target_node,
                1u);
    }
    grid.sync();
    for (std::uint32_t node = tid; node < node_count; node += stride) {
      std::uint32_t incumbents = 0u;
      const std::uint32_t limit =
          node_participation_span_limit(spans, node, node_count);
      const std::uint32_t base = node * kNodeParticipationAperture;
      for (std::uint32_t slot = 0u; slot < limit; ++slot) {
        const NodeCausalParticipation current =
            read_participation_slot(node_next_participation + base + slot);
        if (current.ticket_id != 0ull && current.expiry_tick >= current_tick)
          ++incumbents;
      }
      atomicMax(reinterpret_cast<unsigned long long*>(
                    &metrics->node_participation_uncapped_demand_peak),
                static_cast<unsigned long long>(
                    incumbents + node_candidate_owners[node_count + node]));
    }
    grid.sync();
  }

  for (std::uint32_t i = tid; i < node_slots; i += stride)
    node_candidate_owners[i] = kInvalidIndex;
  grid.sync();

  // Free-slot admit: lowest-index free wins; exit when no pending claims.
  for (std::uint32_t round = 0u; round < kNodeParticipationAperture; ++round) {
    if (tid == 0u) scratch[0] = 0u;
    grid.sync();
    for (std::uint32_t i = tid; i < claim_count; i += stride) {
      if (claims[i].valid != 0u && claims[i].canonical_producer == i &&
          claims[i].eligibility_slot != kInvalidIndex &&
          claims[i].admitted == 0u &&
          claims[i].descriptor.target_node < node_count)
        atomicOr(scratch, 1u);
    }
    grid.sync();
    if (scratch[0] == 0u) break;

    for (std::uint32_t i = tid; i < claim_count; i += stride) {
      if (claims[i].valid == 0u || claims[i].canonical_producer != i ||
          claims[i].eligibility_slot == kInvalidIndex ||
          claims[i].admitted != 0u ||
          claims[i].descriptor.target_node >= node_count)
        continue;
      const std::uint32_t slot =
          claims[i].descriptor.target_node * kNodeParticipationAperture +
          round;
      const NodeCausalParticipation current =
          read_participation_slot(node_next_participation + slot);
      if (participation_slot_reusable(current, current_tick))
        atomicMin(node_candidate_owners + slot, i);
    }
    grid.sync();
    for (std::uint32_t i = tid; i < claim_count; i += stride) {
      if (claims[i].admitted != 0u ||
          claims[i].descriptor.target_node >= node_count)
        continue;
      const std::uint32_t slot =
          claims[i].descriptor.target_node * kNodeParticipationAperture +
          round;
      if (node_candidate_owners[slot] == i)
        claims[i].admitted = round + 1u;
    }
    grid.sync();
  }

  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid == 0u || claims[i].canonical_producer != i ||
        claims[i].eligibility_slot == kInvalidIndex ||
        claims[i].admitted != 0u ||
        claims[i].descriptor.target_node >= node_count)
      continue;
    if (node_next_ancestry_incomplete != nullptr)
      atomicExch(node_next_ancestry_incomplete +
                     claims[i].descriptor.target_node,
                 1u);
    if (metrics != nullptr) {
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->provenance_no_evictable_slot_drops),
                1ULL);
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->node_participation_no_free_drops),
                1ULL);
    }
  }
  grid.sync();

  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid == 0u || claims[i].canonical_producer != i ||
        claims[i].eligibility_slot == kInvalidIndex ||
        claims[i].admitted == 0u ||
        claims[i].admitted > kNodeParticipationAperture)
      continue;
    const std::uint32_t slot =
        claims[i].descriptor.target_node * kNodeParticipationAperture +
        claims[i].admitted - 1u;
    NodeCausalParticipation value =
        read_participation_slot(node_next_participation + slot);
    value.ticket_id = claims[i].descriptor.ticket_id;
    value.expiry_tick = claims[i].descriptor.expiry_tick;
    value.last_refresh_tick = current_tick;
    value.authority = claims[i].descriptor.authority;
    value.authority_incarnation = claims[i].descriptor.authority_incarnation;
    value.claim_incarnation = claims[i].descriptor.claim_incarnation;
    value.current_drive = 1u;
    set_participation_eligibility_tail(
        &value, eligibility_record_ref(claims[i].eligibility_slot,
                                      claims[i].eligibility_generation));
    value.commit_generation =
        publish_participation_slot(node_next_participation + slot, value);
    claims[i].descriptor.eligibility_slot = claims[i].eligibility_slot;
    claims[i].descriptor.eligibility_generation =
        claims[i].eligibility_generation;
    claims[i].descriptor.frozen_eligibility_q16 =
        records[claims[i].eligibility_slot].eligibility_q16;
  }
  grid.sync();

  // Recompute span after publication for capped peak (observer).
  if (metrics != nullptr) {
    compute_node_participation_live_spans(grid, node_next_participation, spans,
                                          node_count, current_tick);
    for (std::uint32_t node = tid; node < node_count; node += stride) {
      std::uint32_t occupied = 0u;
      const std::uint32_t limit =
          node_participation_span_limit(spans, node, node_count);
      const std::uint32_t base = node * kNodeParticipationAperture;
      for (std::uint32_t slot = 0u; slot < limit; ++slot) {
        const NodeCausalParticipation current =
            read_participation_slot(node_next_participation + base + slot);
        if (current.ticket_id != 0ull && current.expiry_tick >= current_tick)
          ++occupied;
      }
      atomicMax(reinterpret_cast<unsigned long long*>(
                    &metrics->node_participation_capped_resident_peak),
                static_cast<unsigned long long>(occupied));
    }
    grid.sync();
  }

  for (std::uint32_t i = tid; i < claim_count; i += stride)
    scan[i] = claims[i].valid != 0u && claims[i].canonical_producer == i &&
                      claims[i].admitted != 0u &&
                      claims[i].admitted <= kNodeParticipationAperture
                  ? 1u
                  : 0u;
  grid.sync();
  const std::uint32_t staged =
      route_transport_exclusive_scan(grid, scan, scratch, claim_count);
  const std::uint32_t base = participation_staging_count != nullptr
                                 ? *participation_staging_count
                                 : 0u;
  const bool staging_fits =
      base <= participation_staging_capacity &&
      staged <= participation_staging_capacity - base;
  for (std::uint32_t i = tid; i < claim_count; i += stride)
    if (scan[i] < staged && claims[i].valid != 0u &&
        claims[i].canonical_producer == i && claims[i].admitted != 0u &&
        claims[i].admitted <= kNodeParticipationAperture && staging_fits)
      participation_staging[base + scan[i]] = claims[i].descriptor;
  grid.sync();
  if (tid == 0u && participation_staging_count != nullptr) {
    if (staging_fits) {
      *participation_staging_count = base + staged;
      if (metrics != nullptr)
        metrics->participation_descriptors_staged += staged;
    } else if (metrics != nullptr) {
      metrics->participation_staging_overflow += staged;
    }
  }
  if (!staging_fits) {
    for (std::uint32_t i = tid; i < claim_count; i += stride)
      if (claims[i].valid != 0u &&
          claims[i].descriptor.target_node < node_count &&
          node_next_ancestry_incomplete != nullptr)
        node_next_ancestry_incomplete[claims[i].descriptor.target_node] = 1u;
  }
  grid.sync();
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_APERTURE_CUH
