#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_DELIVERY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_DELIVERY_CUH
// Mid-include splice for substrate::direct_adult_core via direct_adult_device_ops.cuh.
// Owns eligibility admission + delayed sparse deliver (+ kernel).
// Requires insert/stage/same_action/retire helpers already defined in the parent.
#include "direct_adult_node_participation_aperture.cuh"

struct EligibilityAdmission {
  std::uint32_t slot = kInvalidIndex;
  std::uint32_t generation = 0u;
  std::int32_t canonical_eligibility_q16 = 0;
};

__device__ inline void write_eligibility_batch_claim(
    EligibilityBatchClaim* claims, std::uint32_t capacity,
    std::uint32_t producer, const DirectParticipationDescriptor& descriptor,
    std::int32_t eligibility_q16, std::uint32_t chronology_tick,
    std::uint32_t physical_route_index) {
  if (claims == nullptr || producer >= capacity) return;
  EligibilityBatchClaim claim{};
  claim.descriptor = descriptor;
  claim.eligibility_q16 = eligibility_q16;
  claim.valid = descriptor.ticket_id != 0u &&
                        descriptor.ticket_id != kInvalidTicket &&
                        descriptor.authority !=
                            DirectParticipationAuthority::none &&
                        descriptor.contribution_kind ==
                            DirectContributionKind::sparse_route
                    ? 1u
                    : 0u;
  claim.directory_slot = kInvalidIndex;
  claim.canonical_producer = kInvalidIndex;
  claim.eligibility_slot = kInvalidIndex;
  claim.chronology_tick = chronology_tick;
  claim.physical_route_index = physical_route_index;
  claims[producer] = claim;
}

__device__ inline bool batch_owner_matches_claim(
    std::uint32_t owner, const DirectParticipationDescriptor& claim,
    const EligibilityRecord* records,
    const EligibilityBatchClaim* candidates) {
  if (owner < kMaxLiveEligibilityRecords)
    return eligibility_claim_matches(claim, records[owner]);
  const std::uint32_t candidate = owner - kMaxLiveEligibilityRecords;
  return same_action_claim(claim, candidates[candidate].descriptor);
}

struct EligibilityClaimChronologyLess {
  const EligibilityBatchClaim* claims;
  __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
    if (a == kInvalidIndex) return false;
    if (b == kInvalidIndex) return true;
    if (claims[a].chronology_tick != claims[b].chronology_tick)
      return claims[a].chronology_tick < claims[b].chronology_tick;
    if (claims[a].physical_route_index != claims[b].physical_route_index)
      return claims[a].physical_route_index <
             claims[b].physical_route_index;
    return a < b;
  }
};

__device__ inline bool same_eligibility_priority(
    const EligibilityBatchClaim& left,
    const EligibilityBatchClaim& right) {
  return left.chronology_tick == right.chronology_tick &&
         left.physical_route_index == right.physical_route_index;
}

template <typename Grid>
__device__ inline void deterministic_eligibility_batch(
    Grid grid, EligibilityBatchClaim* claims, std::uint32_t claim_count,
    EligibilityRecord* records, std::uint32_t* live_count,
    std::uint64_t* directory, std::uint32_t* generations,
    std::uint32_t* owners, std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t* free_slots,
    NodeCausalParticipation* node_next_participation,
    std::uint32_t* node_next_ancestry_incomplete,
    std::uint32_t* node_candidate_owners, std::uint32_t node_count,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    std::uint32_t current_tick, AdultCoreMetrics* metrics) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  if (claims == nullptr || records == nullptr || live_count == nullptr ||
      directory == nullptr || generations == nullptr || owners == nullptr ||
      scan == nullptr || scratch == nullptr || free_slots == nullptr ||
      node_candidate_owners == nullptr)
    return;

  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid == 0u) continue;
    if (claims[i].eligibility_q16 <= (kQ16One / 8)) {
      claims[i].valid = 0u;
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->eligibility_threshold_rejects),
                  1ULL);
    } else if (metrics != nullptr) {
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->eligibility_record_write_attempts),
                1ULL);
    }
  }
  grid.sync();

  for (std::uint32_t i = tid; i < kEligibilityClaimDirectoryCapacity;
       i += stride)
    owners[i] = kInvalidIndex;

  // Derive the only full-capacity frontier once. Free ranks are physical-slot
  // ordered, so later allocation is independent of CUDA arrival order. Active
  // record order is irrelevant: each probe uses a physical-slot atomicMin.
  for (std::uint32_t i = tid; i < kMaxLiveEligibilityRecords; i += stride)
    scan[i] =
        (records[i].live == 0u || records[i].expiry_tick < current_tick ||
         records[i].lineage_expiry_tick < current_tick) &&
                generations[i] < ~0u - 1u
            ? 1u
            : 0u;
  grid.sync();
  const std::uint32_t free_count = route_transport_exclusive_scan(
      grid, scan, scratch, kMaxLiveEligibilityRecords);
  for (std::uint32_t i = tid; i < kMaxLiveEligibilityRecords; i += stride)
    if ((records[i].live == 0u || records[i].expiry_tick < current_tick ||
         records[i].lineage_expiry_tick < current_tick) &&
        generations[i] < ~0u - 1u)
      free_slots[scan[i]] = i;
  // Physical-slot order gives the derived open-addressed directory one exact
  // canonical construction. The former parallel atomicMin rounds could leave
  // lawful live records unindexed even though the 2x-capacity table had space;
  // a later batch would then remint the same causal identity. This bounded
  // 16K metadata pass removes grid-wide collision rounds while the much larger
  // candidate and propagation work remains parallel.
  if (tid == 0u) {
    *live_count = 0u;
    for (std::uint32_t record_slot = 0u;
         record_slot < kMaxLiveEligibilityRecords; ++record_slot) {
      EligibilityRecord& record = records[record_slot];
      if (record.live == 0u || record.expiry_tick < current_tick ||
          record.lineage_expiry_tick < current_tick)
        continue;
      const DirectParticipationDescriptor descriptor =
          eligibility_record_claim(record);
      bool indexed = false;
      for (std::uint32_t probe = 0u;
           probe < kEligibilityClaimDirectoryCapacity; ++probe) {
        const std::uint32_t directory_slot =
            (static_cast<std::uint32_t>(
                 eligibility_claim_hash(descriptor)) +
             probe) &
            (kEligibilityClaimDirectoryCapacity - 1u);
        const std::uint32_t owner = owners[directory_slot];
        if (owner == kInvalidIndex) {
          owners[directory_slot] = record_slot;
          indexed = true;
          break;
        }
        if (owner < kMaxLiveEligibilityRecords &&
            eligibility_record_identity_matches(record, records[owner])) {
          indexed = true;
          break;
        }
      }
      if (!indexed) {
        if (record.target_node < node_count &&
            node_next_ancestry_incomplete != nullptr)
          node_next_ancestry_incomplete[record.target_node] = 1u;
        if (metrics != nullptr) ++metrics->eligibility_capacity_rejects;
      }
      ++*live_count;
    }
  }
  grid.sync();

  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    claims[i].directory_slot = kInvalidIndex;
    claims[i].canonical_producer = kInvalidIndex;
    claims[i].eligibility_slot = kInvalidIndex;
    claims[i].eligibility_generation = 0u;
    claims[i].admitted = 0u;
  }
  grid.sync();

  if (tid == 0u) node_candidate_owners[0] = 0u;
  grid.sync();
  for (std::uint32_t i = tid; i < claim_count; i += stride)
    if (claims[i].valid != 0u)
      scan[atomicAdd(node_candidate_owners, 1u)] = i;
  grid.sync();
  std::uint32_t unresolved = node_candidate_owners[0];
  std::uint32_t* unresolved_now = scan;
  std::uint32_t* unresolved_next = scratch;
  for (std::uint32_t probe = 0u;
       probe < kEligibilityClaimMaxProbes && unresolved != 0u; ++probe) {
    for (std::uint32_t rank = tid; rank < unresolved; rank += stride) {
      const std::uint32_t i = unresolved_now[rank];
      const std::uint32_t slot =
          (static_cast<std::uint32_t>(
               eligibility_claim_hash(claims[i].descriptor)) +
           probe) &
          (kEligibilityClaimDirectoryCapacity - 1u);
      const std::uint32_t owner = owners[slot];
      if (owner != kInvalidIndex &&
          batch_owner_matches_claim(
              owner, claims[i].descriptor, records, claims)) {
        claims[i].directory_slot = slot;
        continue;
      }
      if (owner == kInvalidIndex)
        atomicMin(owners + slot, kMaxLiveEligibilityRecords + i);
    }
    grid.sync();
    if (tid == 0u) node_candidate_owners[0] = 0u;
    grid.sync();
    for (std::uint32_t rank = tid; rank < unresolved; rank += stride) {
      const std::uint32_t i = unresolved_now[rank];
      const std::uint32_t slot =
          (static_cast<std::uint32_t>(
               eligibility_claim_hash(claims[i].descriptor)) +
           probe) &
          (kEligibilityClaimDirectoryCapacity - 1u);
      const std::uint32_t owner = owners[slot];
      if (owner == kMaxLiveEligibilityRecords + i ||
          (owner != kInvalidIndex &&
           batch_owner_matches_claim(
               owner, claims[i].descriptor, records, claims))) {
        claims[i].directory_slot = slot;
      } else {
        const std::uint32_t next_rank =
            atomicAdd(node_candidate_owners, 1u);
        unresolved_next[next_rank] = i;
      }
    }
    grid.sync();
    unresolved = node_candidate_owners[0];
    std::uint32_t* swap = unresolved_now;
    unresolved_now = unresolved_next;
    unresolved_next = swap;
  }

  for (std::uint32_t rank = tid; rank < unresolved; rank += stride) {
    const std::uint32_t i = unresolved_now[rank];
    if (claims[i].descriptor.target_node < node_count &&
        node_next_ancestry_incomplete != nullptr)
      atomicExch(node_next_ancestry_incomplete +
                     claims[i].descriptor.target_node,
                 1u);
    claims[i].valid = 0u;
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->eligibility_capacity_rejects),
                1ULL);
  }
  grid.sync();

  // Pick one canonical physical producer for each exact claim identity.
  for (std::uint32_t i = tid; i < kEligibilityClaimDirectoryCapacity;
       i += stride)
    directory[i] = ~0ULL;
  grid.sync();
  for (std::uint32_t i = tid; i < claim_count; i += stride)
    if (claims[i].valid != 0u &&
        claims[i].directory_slot != kInvalidIndex)
      atomicMin(reinterpret_cast<unsigned long long*>(
                    directory + claims[i].directory_slot),
                static_cast<unsigned long long>(i) + 1ULL);
  grid.sync();
  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid == 0u ||
        claims[i].directory_slot == kInvalidIndex)
      continue;
    claims[i].canonical_producer =
        directory[claims[i].directory_slot] == ~0ULL
            ? kInvalidIndex
            : static_cast<std::uint32_t>(
                  directory[claims[i].directory_slot] - 1u);
    const std::uint32_t leader = claims[i].canonical_producer;
    if (leader < claim_count) {
      atomicMax(&claims[leader].descriptor.expiry_tick,
                claims[i].descriptor.expiry_tick);
      atomicMax(&claims[leader].descriptor.lineage_expiry_tick,
                claims[i].descriptor.lineage_expiry_tick);
      atomicMax(reinterpret_cast<int*>(&claims[leader].eligibility_q16),
                claims[i].eligibility_q16);
    }
  }
  grid.sync();

  for (std::uint32_t i = tid; i < claim_count; i += stride)
    scan[i] = claims[i].valid != 0u &&
                      claims[i].canonical_producer == i
                  ? 1u
                  : 0u;
  grid.sync();
  const std::uint32_t canonical_count =
      route_transport_exclusive_scan(grid, scan, scratch, claim_count);
  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid != 0u && claims[i].canonical_producer == i)
      scratch[scan[i]] = i;
    claims[i].admitted = 0u;
  }
  grid.sync();
  route_transport_bitonic_sort_indices(
      grid, scratch, canonical_count, claim_count,
      EligibilityClaimChronologyLess{claims});
  for (std::uint32_t rank = tid; rank < canonical_count; rank += stride) {
    const std::uint32_t claim = scratch[rank];
    const bool group_start =
        rank == 0u ||
        !same_eligibility_priority(claims[scratch[rank - 1u]],
                                   claims[claim]);
    std::uint32_t new_in_group = 0u;
    if (group_start) {
      std::uint32_t end = rank;
      while (end < canonical_count &&
             same_eligibility_priority(claims[claim],
                                       claims[scratch[end]]))
        {
          const std::uint32_t candidate = scratch[end++];
          if (owners[claims[candidate].directory_slot] ==
              kMaxLiveEligibilityRecords + candidate)
            ++new_in_group;
        }
    }
    scan[rank] = new_in_group;
  }
  grid.sync();
  route_transport_exclusive_scan(grid, scan, node_candidate_owners,
                                 canonical_count);
  for (std::uint32_t rank = tid; rank < canonical_count; rank += stride) {
    std::uint32_t group_start = rank;
    while (group_start != 0u &&
           same_eligibility_priority(claims[scratch[group_start - 1u]],
                                     claims[scratch[rank]]))
      --group_start;
    std::uint32_t group_end = rank + 1u;
    while (group_end < canonical_count &&
           same_eligibility_priority(claims[scratch[rank]],
                                     claims[scratch[group_end]]))
      ++group_end;
    const std::uint32_t base = scan[group_start];
    std::uint32_t group_new = 0u;
    std::uint32_t new_before = 0u;
    for (std::uint32_t cursor = group_start; cursor < group_end; ++cursor) {
      const std::uint32_t candidate = scratch[cursor];
      const bool is_new = owners[claims[candidate].directory_slot] ==
                          kMaxLiveEligibilityRecords + candidate;
      group_new += is_new ? 1u : 0u;
      if (cursor < rank) new_before += is_new ? 1u : 0u;
    }
    const bool fits = base <= free_count && group_new <= free_count - base;
    const std::uint32_t claim = scratch[rank];
    if (fits) {
      if (owners[claims[claim].directory_slot] ==
          kMaxLiveEligibilityRecords + claim)
        claims[claim].admitted = base + new_before + 1u;
    } else {
      claims[claim].valid = 0u;
      if (claims[claim].descriptor.target_node < node_count &&
          node_next_ancestry_incomplete != nullptr)
        atomicExch(node_next_ancestry_incomplete +
                       claims[claim].descriptor.target_node,
                   1u);
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->eligibility_capacity_rejects),
                  1ULL);
    }
  }
  grid.sync();

  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    if (claims[i].valid == 0u || claims[i].canonical_producer != i)
      continue;
    const std::uint32_t owner = owners[claims[i].directory_slot];
    if (owner < kMaxLiveEligibilityRecords) {
      EligibilityRecord& record = records[owner];
      atomicMax(&record.expiry_tick, claims[i].descriptor.expiry_tick);
      atomicMax(&record.lineage_expiry_tick,
                claims[i].descriptor.lineage_expiry_tick);
      atomicMax(reinterpret_cast<int*>(&record.eligibility_q16),
                claims[i].eligibility_q16);
      claims[i].eligibility_slot = owner;
      claims[i].eligibility_generation = generations[owner];
    } else if (claims[i].admitted != 0u) {
      const std::uint32_t slot =
          free_slots[claims[i].admitted - 1u];
      if (records[slot].live != 0u &&
          (records[slot].expiry_tick < current_tick ||
           records[slot].lineage_expiry_tick < current_tick) &&
          metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->eligibility_expired_records_reclaimed),
                  1ULL);
      EligibilityRecord record{};
      record.ticket_id = claims[i].descriptor.ticket_id;
      record.source_node = claims[i].descriptor.source_node;
      record.target_node = claims[i].descriptor.target_node;
      record.route_index = claims[i].descriptor.route_index;
      record.context_signature = claims[i].descriptor.context_signature;
      record.eligibility_q16 = claims[i].eligibility_q16;
      record.expiry_tick = claims[i].descriptor.expiry_tick;
      record.live = 1u;
      record.claim_incarnation =
          claims[i].descriptor.claim_incarnation;
      record.route_incarnation =
          claims[i].descriptor.route_incarnation;
      record.authority = claims[i].descriptor.authority;
      record.authority_incarnation =
          claims[i].descriptor.authority_incarnation;
      record.parent_eligibility_ref =
          claims[i].descriptor.parent_eligibility_ref;
      record.lineage_expiry_tick =
          claims[i].descriptor.lineage_expiry_tick;
      record.ancestry_depth = claims[i].descriptor.ancestry_depth;
      const std::uint32_t generation = generations[slot] + 1u;
      records[slot] = record;
      generations[slot] = generation;
      claims[i].eligibility_slot = slot;
      claims[i].eligibility_generation = generation;
      atomicAdd(live_count, 1u);
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->eligibility_records_written),
                  1ULL);
    }
  }
  grid.sync();
  for (std::uint32_t i = tid; i < claim_count; i += stride) {
    const std::uint32_t leader = claims[i].canonical_producer;
    if (claims[i].valid == 0u || leader >= claim_count) continue;
    if (claims[leader].valid == 0u) {
      claims[i].valid = 0u;
      continue;
    }
    claims[i].eligibility_slot = claims[leader].eligibility_slot;
    claims[i].eligibility_generation =
        claims[leader].eligibility_generation;
  }
  grid.sync();

  // Publish the derived lookup only after every durable record is complete.
  for (std::uint32_t i = tid; i < kEligibilityClaimDirectoryCapacity;
       i += stride) {
    const std::uint32_t owner = owners[i];
    std::uint32_t slot = kInvalidIndex;
    std::uint32_t generation = 0u;
    if (owner < kMaxLiveEligibilityRecords) {
      slot = owner;
      generation = generations[slot];
    } else if (owner != kInvalidIndex) {
      const std::uint32_t candidate = owner - kMaxLiveEligibilityRecords;
      if (candidate < claim_count) {
        slot = claims[candidate].eligibility_slot;
        generation = claims[candidate].eligibility_generation;
      }
    }
    directory[i] =
        slot < kMaxLiveEligibilityRecords && generation != 0u
            ? (static_cast<std::uint64_t>(generation) << 32) |
                  (static_cast<std::uint64_t>(slot) + 1u)
            : 0u;
  }
  grid.sync();

  admit_node_participation_aperture(
      grid, claims, claim_count, records, scan, scratch,
      node_next_participation, node_next_ancestry_incomplete,
      node_candidate_owners, node_count, participation_staging,
      participation_staging_count, participation_staging_capacity,
      current_tick, metrics);
}

template <typename Grid>
__device__ inline void derive_ingress_eligibility_free_frontier(
    Grid grid, EligibilityRecord* records, std::uint32_t* generations,
    std::uint32_t* live_count, std::uint32_t current_tick,
    std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t* free_slots, AdultCoreMetrics* metrics) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  if (records == nullptr || generations == nullptr || live_count == nullptr ||
      scan == nullptr || scratch == nullptr || free_slots == nullptr)
    return;
  if (tid == 0u) *live_count = 0u;
  grid.sync();
  for (std::uint32_t i = tid; i < kMaxLiveEligibilityRecords; i += stride) {
    const bool active = records[i].live != 0u &&
                        records[i].expiry_tick >= current_tick &&
                        records[i].lineage_expiry_tick >= current_tick;
    if (active) {
      atomicAdd(live_count, 1u);
    } else {
      if (records[i].live != 0u && metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->eligibility_expired_records_reclaimed),
                  1ULL);
      records[i].live = 0u;
    }
    scan[i] = !active && generations[i] < ~0u - 1u ? 1u : 0u;
  }
  grid.sync();
  const std::uint32_t free_count = route_transport_exclusive_scan(
      grid, scan, scratch, kMaxLiveEligibilityRecords);
  for (std::uint32_t i = tid; i < kMaxLiveEligibilityRecords; i += stride)
    if (records[i].live == 0u && generations[i] < ~0u - 1u)
      free_slots[scan[i]] = i;
  grid.sync();
  if (tid == 0u) {
    scan[0] = free_count;
    scan[1] = 0u;
  }
  grid.sync();
}

__device__ inline EligibilityAdmission admit_ingress_eligibility_claim(
    const DirectParticipationDescriptor& claim,
    std::int32_t eligibility_q16, EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* claim_directory,
    std::uint32_t* record_generations,
    const std::uint32_t* free_slots,
    std::uint32_t* free_state,
    std::uint32_t current_tick, AdultCoreMetrics* metrics) {
  if (claim.contribution_kind != DirectContributionKind::sparse_route ||
      claim.authority != DirectParticipationAuthority::independent_external ||
      claim.parent_eligibility_ref != 0u || claim.ancestry_depth != 1u ||
      claim.lineage_expiry_tick < current_tick ||
      eligibility_table == nullptr || live_eligibility_count == nullptr ||
      claim_directory == nullptr || record_generations == nullptr ||
      free_slots == nullptr || free_state == nullptr)
    return {};
  if (eligibility_q16 <= (kQ16One / 8)) {
    if (metrics != nullptr) ++metrics->eligibility_threshold_rejects;
    return {};
  }
  if (metrics != nullptr) ++metrics->eligibility_record_write_attempts;

  const std::uint64_t hash = eligibility_claim_hash(claim);
  const std::uint32_t start = static_cast<std::uint32_t>(hash) &
                              (kEligibilityClaimDirectoryCapacity - 1u);
  std::uint32_t directory_slot = kInvalidIndex;
  for (std::uint32_t probe = 0u; probe < kEligibilityClaimMaxProbes; ++probe) {
    const std::uint32_t index =
        (start + probe) & (kEligibilityClaimDirectoryCapacity - 1u);
    const std::uint64_t packed = claim_directory[index];
    if (packed == 0u) {
      if (directory_slot == kInvalidIndex) directory_slot = index;
      continue;
    }
    if (packed == ~0ULL) return {};
    const std::uint32_t slot = static_cast<std::uint32_t>(packed) - 1u;
    const std::uint32_t generation = static_cast<std::uint32_t>(packed >> 32u);
    const bool current = slot < kMaxLiveEligibilityRecords &&
                         generation != 0u &&
                         record_generations[slot] == generation;
    if (!current) {
      if (directory_slot == kInvalidIndex) directory_slot = index;
      continue;
    }
    EligibilityRecord& record = eligibility_table[slot];
    if (record.live != 0u && record.expiry_tick >= current_tick &&
        record.lineage_expiry_tick >= current_tick &&
        eligibility_claim_matches(claim, record)) {
      if (claim.expiry_tick > record.expiry_tick)
        record.expiry_tick = claim.expiry_tick;
      if (claim.lineage_expiry_tick > record.lineage_expiry_tick)
        record.lineage_expiry_tick = claim.lineage_expiry_tick;
      if (eligibility_q16 > record.eligibility_q16)
        record.eligibility_q16 = eligibility_q16;
      return EligibilityAdmission{slot, generation, record.eligibility_q16};
    }
    if ((record.live == 0u || record.expiry_tick < current_tick ||
         record.lineage_expiry_tick < current_tick) &&
        directory_slot == kInvalidIndex)
      directory_slot = index;
  }

  const std::uint32_t free_count = free_state[0];
  const std::uint32_t free_cursor = free_state[1];
  if (directory_slot == kInvalidIndex || free_cursor >= free_count) {
    if (metrics != nullptr) ++metrics->eligibility_capacity_rejects;
    return {};
  }
  const std::uint32_t slot = free_slots[free_cursor];
  if (slot >= kMaxLiveEligibilityRecords ||
      record_generations[slot] >= ~0u - 1u ||
      (eligibility_table[slot].live != 0u &&
       eligibility_table[slot].expiry_tick >= current_tick &&
       eligibility_table[slot].lineage_expiry_tick >= current_tick)) {
    if (metrics != nullptr) ++metrics->eligibility_capacity_rejects;
    return {};
  }

  EligibilityRecord record{};
  record.ticket_id = claim.ticket_id;
  record.source_node = claim.source_node;
  record.target_node = claim.target_node;
  record.route_index = claim.route_index;
  record.context_signature = claim.context_signature;
  record.eligibility_q16 = eligibility_q16;
  record.expiry_tick = claim.expiry_tick;
  record.live = 1u;
  record.claim_incarnation = claim.claim_incarnation;
  record.route_incarnation = claim.route_incarnation;
  record.authority = claim.authority;
  record.authority_incarnation = claim.authority_incarnation;
  record.parent_eligibility_ref = claim.parent_eligibility_ref;
  record.lineage_expiry_tick = claim.lineage_expiry_tick;
  record.ancestry_depth = claim.ancestry_depth;
  const std::uint32_t generation = record_generations[slot] + 1u;
  eligibility_table[slot] = record;
  record_generations[slot] = generation;
  claim_directory[directory_slot] =
      (static_cast<std::uint64_t>(generation) << 32u) |
      (static_cast<std::uint64_t>(slot) + 1u);
  free_state[1] = free_cursor + 1u;
  ++*live_eligibility_count;
  if (metrics != nullptr) ++metrics->eligibility_records_written;
  return EligibilityAdmission{slot, generation, eligibility_q16};
}

__device__ inline void prepare_delayed_sparse_packet_batch(
    DelayedSparsePacket* packets, DelayedPacketIdentity* identities,
    DirectNode* nodes,
    DirectRoute* routes, const std::uint64_t* route_incarnations,
    std::uint64_t* route_opportunity_incarnations,
    std::uint32_t node_count, std::uint32_t route_capacity,
    std::int32_t* node_incoming_excitation,
    EligibilityRecord* eligibility_table,
    std::uint32_t* eligibility_record_generations,
    EligibilityBatchClaim* claims, std::uint32_t claim_capacity,
    std::uint32_t* node_next_ancestry_incomplete,
    std::uint32_t packet_index, std::uint32_t claim_producer,
    std::uint32_t current_tick,
    std::int32_t eligibility_decay_q16, std::uint32_t horizon_ticks,
    AdultCoreMetrics* metrics) {
  if (packets == nullptr || packet_index >= kMaxDelayedSparsePackets ||
      atomicAdd(&packets[packet_index].live, 0u) !=
          static_cast<std::uint32_t>(DelayedPacketSlotState::published) ||
      packets[packet_index].due_tick > current_tick)
    return;
  const DelayedSparsePacket packet = packets[packet_index];
  packets[packet_index].live =
      static_cast<std::uint32_t>(DelayedPacketSlotState::reserved);
  const bool route_current =
      nodes != nullptr && routes != nullptr &&
      route_incarnations != nullptr && packet.source_node < node_count &&
      packet.target_node < node_count &&
      packet.route_index < route_capacity &&
      route_incarnations[packet.route_index] ==
          packet.route_incarnation &&
      routes[packet.route_index].source == packet.source_node &&
      routes[packet.route_index].target == packet.target_node &&
      (routes[packet.route_index].flags &
       direct_network::kRouteFlagActive) != 0u;
  if (!route_current) {
    retire_delayed_packet(packets, identities, packet_index);
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->delayed_packets_stale_rejected),
                1ULL);
    return;
  }
  DirectRoute& route = routes[packet.route_index];
  if (packet.drive_q16 != 0 &&
      route_opportunity_incarnations != nullptr)
    route_opportunity_incarnations[packet.route_index] =
        packet.route_incarnation;
  if (node_incoming_excitation != nullptr)
    atomicAdd(&node_incoming_excitation[packet.target_node],
              packet.drive_q16);
  const std::int32_t arrival_target =
      nodes[packet.target_node].activation_q16;
  const std::int32_t dose = mul_q16(
      packet.source_activation_q16,
      arrival_target > 0 ? arrival_target : kQ16Half);
  const std::int32_t eligibility = clamp_q16(
      mul_q16(route.eligibility_q16, eligibility_decay_q16) + dose,
      0, 4 * kQ16One);
  route.eligibility_q16 = eligibility;

  bool all_claims_lawful = true;
  const std::uint32_t base =
      claim_producer * kRouteTransportProducerWidth;
  for (std::uint32_t s = 0u;
       s < packet.claim_count && s < kMaxProvenanceSlotsPerNode; ++s) {
    if (packet.drive_q16 == 0) break;
    const DirectParticipationDescriptor source = packet.source_claims[s];
    if (source.ticket_id == 0u || source.ticket_id == kInvalidTicket) {
      all_claims_lawful = false;
      continue;
    }
    DirectParticipationDescriptor descriptor{};
    descriptor.ticket_id = source.ticket_id;
    descriptor.source_node = packet.source_node;
    descriptor.target_node = packet.target_node;
    descriptor.route_index = packet.route_index;
    descriptor.context_signature = packet.context_signature;
    descriptor.expiry_tick = current_tick + horizon_ticks;
    descriptor.claim_incarnation = source.claim_incarnation;
    descriptor.route_incarnation = packet.route_incarnation;
    descriptor.authority = source.authority;
    descriptor.authority_incarnation = source.authority_incarnation;
    descriptor.contribution_kind = DirectContributionKind::sparse_route;
    descriptor.parent_eligibility_ref =
        source.parent_eligibility_ref;
    if (!inherit_eligibility_lineage(
            &descriptor, source.expiry_tick, eligibility_table,
            eligibility_record_generations, current_tick)) {
      all_claims_lawful = false;
      if (node_next_ancestry_incomplete != nullptr)
        atomicExch(node_next_ancestry_incomplete + packet.target_node,
                   1u);
      continue;
    }
    write_eligibility_batch_claim(
        claims, claim_capacity, base + s, descriptor, eligibility,
        packet.due_tick, packet.route_index);
  }
  if (packet.drive_q16 != 0 &&
      (packet.source_ancestry_incomplete != 0u ||
       (packet.claim_count != 0u && !all_claims_lawful)) &&
      node_next_ancestry_incomplete != nullptr)
    atomicExch(node_next_ancestry_incomplete + packet.target_node, 1u);
  retire_delayed_packet(packets, identities, packet_index);
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->delayed_packets_delivered),
              1ULL);
}

template <typename Grid>
__device__ inline void execute_deterministic_route_transport(
    Grid grid, DirectNode* nodes, DirectRoute* routes,
    const std::uint64_t* route_incarnations,
    std::uint64_t* route_opportunity_incarnations,
    std::uint32_t node_count, std::uint32_t route_capacity,
    std::int32_t* node_incoming_excitation,
    DelayedSparsePacket* delayed_packets,
    std::uint32_t* delayed_packet_live_count,
    std::uint32_t* delayed_packet_free_head,
    std::uint32_t* delayed_packet_next_free,
    DelayedPacketIdentity* delayed_packet_identities,
    const NodeCausalParticipation* node_active_participation,
    NodeCausalParticipation* node_next_participation,
    const std::uint32_t* node_active_ancestry_incomplete,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    RouteTransportPhaseView transport, std::uint32_t current_tick,
    std::int32_t eligibility_decay_q16, std::uint32_t horizon_ticks,
    std::uint32_t honor_inhibitory_sign, AdultCoreMetrics* metrics) {
  const std::uint32_t tid = grid.thread_rank();
  const std::uint32_t stride = grid.size();
  const std::uint32_t active_packet_count = *delayed_packet_live_count;
  for (std::uint32_t rank = tid; rank < active_packet_count; rank += stride) {
    const std::uint32_t packet = delayed_packet_next_free[rank];
    transport.scan_a[rank] =
        packet < kMaxDelayedSparsePackets &&
                delayed_packets[packet].live ==
                    static_cast<std::uint32_t>(
                        DelayedPacketSlotState::published) &&
                delayed_packets[packet].due_tick <= current_tick
            ? 1u
            : 0u;
  }
  grid.sync();
  const std::uint32_t due_count = route_transport_exclusive_scan(
      grid, transport.scan_a, transport.scan_b,
      active_packet_count);
  for (std::uint32_t rank = tid; rank < active_packet_count; rank += stride) {
    const std::uint32_t packet = delayed_packet_next_free[rank];
    if (packet < kMaxDelayedSparsePackets &&
        delayed_packets[packet].live ==
            static_cast<std::uint32_t>(
                DelayedPacketSlotState::published) &&
        delayed_packets[packet].due_tick <= current_tick)
      transport.scan_b[transport.scan_a[rank]] = packet;
  }
  grid.sync();
  route_transport_bitonic_sort_indices(
      grid, transport.scan_b, due_count, transport.scan_capacity,
      DelayedPacketChronologyLess{delayed_packets});
  const std::uint32_t delayed_claim_count =
      due_count * kRouteTransportProducerWidth;
  for (std::uint32_t i = tid; i < delayed_claim_count; i += stride)
    transport.claims[i] = EligibilityBatchClaim{};
  grid.sync();
  for (std::uint32_t rank = tid; rank < due_count; rank += stride) {
    const std::uint32_t packet_index = transport.scan_b[rank];
    if (rank != 0u &&
        delayed_packets[transport.scan_b[rank - 1u]].route_index ==
            delayed_packets[packet_index].route_index)
      continue;
    std::uint32_t end = rank + 1u;
    while (end < due_count &&
           delayed_packets[transport.scan_b[end]].route_index ==
               delayed_packets[packet_index].route_index)
      ++end;
    for (std::uint32_t cursor = rank; cursor < end;) {
      const DelayedSparsePacket first =
          delayed_packets[transport.scan_b[cursor]];
      std::uint32_t duplicate_end = cursor + 1u;
      while (duplicate_end < end) {
        const DelayedSparsePacket next =
            delayed_packets[transport.scan_b[duplicate_end]];
        if (next.due_tick != first.due_tick ||
            next.route_incarnation != first.route_incarnation)
          break;
        ++duplicate_end;
      }
      if (duplicate_end != cursor + 1u) {
        for (std::uint32_t duplicate = cursor;
             duplicate < duplicate_end; ++duplicate) {
          const std::uint32_t rejected = transport.scan_b[duplicate];
          const DelayedSparsePacket packet = delayed_packets[rejected];
          if (packet.drive_q16 != 0 &&
              (packet.source_ancestry_incomplete != 0u ||
               packet.claim_count != 0u) &&
              node_next_ancestry_incomplete != nullptr)
            atomicExch(node_next_ancestry_incomplete + packet.target_node,
                       1u);
          retire_delayed_packet(
              delayed_packets, delayed_packet_identities, rejected);
        }
        if (metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->delayed_packets_duplicate_rejections),
                    static_cast<unsigned long long>(duplicate_end - cursor));
      } else {
        prepare_delayed_sparse_packet_batch(
            delayed_packets, delayed_packet_identities, nodes, routes,
            route_incarnations,
            route_opportunity_incarnations, node_count, route_capacity,
            node_incoming_excitation, eligibility_table,
            eligibility_record_generations, transport.claims,
            transport.claim_capacity, node_next_ancestry_incomplete,
            transport.scan_b[cursor], cursor, current_tick,
            eligibility_decay_q16, horizon_ticks, metrics);
      }
      cursor = duplicate_end;
    }
  }
  grid.sync();
  compact_delayed_packet_frontier(
      grid, delayed_packets, active_packet_count,
      delayed_packet_live_count, delayed_packet_free_head,
      delayed_packet_next_free, transport.scan_a, transport.scan_b,
      transport.free_slots);
  deterministic_eligibility_batch(
      grid, transport.claims, delayed_claim_count, eligibility_table,
      live_eligibility_count, eligibility_claim_directory,
      eligibility_record_generations, transport.eligibility_owners,
      transport.scan_a, transport.scan_b, transport.free_slots,
      node_next_participation, node_next_ancestry_incomplete,
      transport.node_candidate_owners, node_count, participation_staging,
      participation_staging_count, participation_staging_capacity,
      current_tick, metrics);

  const std::uint32_t immediate_claim_count =
      route_capacity * kRouteTransportProducerWidth;
  for (std::uint32_t i = tid; i < immediate_claim_count; i += stride)
    transport.claims[i] = EligibilityBatchClaim{};
  for (std::uint32_t route_index = tid; route_index < route_capacity;
       route_index += stride)
    transport.proposals[route_index] = RouteTransportProposal{};
  // Live high-water over active frontier — pack/propose scan only [0,span).
  compute_node_participation_live_spans(
      grid, node_active_participation, transport.node_candidate_owners,
      node_count, current_tick);
  grid.sync();

  for (std::uint32_t route_index = tid; route_index < route_capacity;
       route_index += stride) {
    DirectRoute& route = routes[route_index];
    if ((route.flags & direct_network::kRouteFlagActive) == 0u ||
        route.source >= node_count || route.target >= node_count)
      continue;
    const DirectNode source = nodes[route.source];
    if (source.activation_q16 <= 0 ||
        route_index < source.route_offset ||
        route_index - source.route_offset >=
            bounded_route_scan_count(source))
      continue;
    const std::int32_t drive = signed_sparse_route_delivery_q16(
        source, route, competition_output_gain_q16(source),
        honor_inhibitory_sign != 0u,
        direct_network::direct_tube_chemistry_q16(
            source.chemotype, nodes[route.target].chemotype)
            .conductance_gain_q16);
    const std::uint32_t context =
        route.eligibility_context ^
        (route.source * 2654435761U);
    const bool source_incomplete =
        node_active_ancestry_incomplete != nullptr &&
        node_active_ancestry_incomplete[route.source] != 0u;
    if (route.delay != 0u) {
      propose_delayed_sparse_packet(
          transport.proposals, route_capacity,
          node_active_participation, eligibility_table,
          eligibility_record_generations,
          node_next_ancestry_incomplete != nullptr
              ? node_next_ancestry_incomplete + route.target
              : nullptr,
          route.source, route.target,
          route_index,
          route_incarnations != nullptr
              ? route_incarnations[route_index]
              : 0u,
          context, source.activation_q16, drive,
          source_incomplete ? 1u : 0u,
          current_tick + route.delay, current_tick,
          transport.node_candidate_owners, node_count);
      continue;
    }
    if (drive != 0 && route_opportunity_incarnations != nullptr &&
        route_incarnations != nullptr)
      route_opportunity_incarnations[route_index] =
          route_incarnations[route_index];
    if (node_incoming_excitation != nullptr)
      atomicAdd(node_incoming_excitation + route.target, drive);
    const std::int32_t target_activation =
        nodes[route.target].activation_q16;
    const std::int32_t dose = mul_q16(
        source.activation_q16,
        target_activation > 0 ? target_activation : kQ16Half);
    const std::int32_t eligibility = clamp_q16(
        mul_q16(route.eligibility_q16, eligibility_decay_q16) + dose,
        0, 4 * kQ16One);
    route.eligibility_q16 = eligibility;
    bool source_claim = false;
    bool all_claims_lawful = true;
    std::uint32_t packed = 0u;
    const std::uint32_t source_span = node_participation_span_limit(
        transport.node_candidate_owners, route.source, node_count);
    for (std::uint32_t s = 0u; s < source_span; ++s) {
      const NodeCausalParticipation participant = read_participation_slot(
          node_active_participation +
          route.source * kNodeParticipationAperture + s);
      if (participant.ticket_id == 0u ||
          participant.ticket_id == kInvalidTicket ||
          participant.current_drive == 0u ||
          participant.expiry_tick < current_tick)
        continue;
      source_claim = true;
      if (drive == 0) continue;
      if (participant.authority == DirectParticipationAuthority::none) {
        all_claims_lawful = false;
        continue;
      }
      if (packed >= kRouteTransportProducerWidth) {
        all_claims_lawful = false;
        break;
      }
      DirectParticipationDescriptor descriptor{};
      descriptor.ticket_id = participant.ticket_id;
      descriptor.source_node = route.source;
      descriptor.target_node = route.target;
      descriptor.route_index = route_index;
      descriptor.context_signature = context;
      descriptor.expiry_tick = participant.expiry_tick;
      descriptor.claim_incarnation =
          participant.claim_incarnation;
      descriptor.route_incarnation =
          route_incarnations != nullptr
              ? route_incarnations[route_index]
              : 0u;
      descriptor.authority = participant.authority;
      descriptor.authority_incarnation =
          participant.authority_incarnation;
      descriptor.contribution_kind =
          DirectContributionKind::sparse_route;
      descriptor.parent_eligibility_ref =
          participation_eligibility_tail(participant);
      if (!inherit_eligibility_lineage(
              &descriptor, participant.expiry_tick, eligibility_table,
              eligibility_record_generations, current_tick)) {
        all_claims_lawful = false;
        continue;
      }
      write_eligibility_batch_claim(
          transport.claims, transport.claim_capacity,
          route_index * kRouteTransportProducerWidth + packed, descriptor,
          eligibility, current_tick, route_index);
      ++packed;
    }
    if (drive != 0 &&
        (source_incomplete || !source_claim || !all_claims_lawful) &&
        node_next_ancestry_incomplete != nullptr)
      atomicExch(node_next_ancestry_incomplete + route.target, 1u);
  }
  grid.sync();

  materialize_route_transport_proposals(
      grid, transport.proposals, route_capacity, current_tick, node_count,
      transport.cursor, delayed_packets, delayed_packet_live_count,
      delayed_packet_free_head, delayed_packet_next_free,
      delayed_packet_identities, transport.scan_a, transport.scan_b,
      transport.scan_capacity, transport.free_slots,
      transport.eligibility_owners,
      node_next_ancestry_incomplete, metrics);
  deterministic_eligibility_batch(
      grid, transport.claims, immediate_claim_count, eligibility_table,
      live_eligibility_count, eligibility_claim_directory,
      eligibility_record_generations, transport.eligibility_owners,
      transport.scan_a, transport.scan_b, transport.free_slots,
      node_next_participation, node_next_ancestry_incomplete,
      transport.node_candidate_owners, node_count, participation_staging,
      participation_staging_count, participation_staging_capacity,
      current_tick, metrics);
  // Transient phase work is never checkpoint identity.
  for (std::uint32_t i = tid; i < transport.claim_capacity; i += stride)
    transport.claims[i] = EligibilityBatchClaim{};
  for (std::uint32_t i = tid; i < route_capacity; i += stride)
    transport.proposals[i] = RouteTransportProposal{};
  for (std::uint32_t i = tid; i < transport.scan_capacity; i += stride) {
    transport.scan_a[i] = 0u;
    transport.scan_b[i] = 0u;
  }
  constexpr std::uint32_t free_capacity =
      kMaxLiveEligibilityRecords > kMaxDelayedSparsePackets
          ? kMaxLiveEligibilityRecords
          : kMaxDelayedSparsePackets;
  for (std::uint32_t i = tid; i < free_capacity; i += stride)
    transport.free_slots[i] = 0u;
  for (std::uint32_t i = tid; i < kRouteTransportOwnerScratchCapacity;
       i += stride)
    transport.eligibility_owners[i] = 0u;
  const std::uint64_t node_owner_capacity =
      static_cast<std::uint64_t>(node_count) *
      kNodeParticipationAperture;
  const std::uint64_t owner_clear_capacity =
      node_owner_capacity > transport.claim_capacity
          ? node_owner_capacity
          : transport.claim_capacity;
  for (std::uint64_t i = tid; i < owner_clear_capacity; i += stride)
    transport.node_candidate_owners[i] = 0u;
  grid.sync();
}

// Temporary executor compatibility: history is never expression authority.

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_SPARSE_DELIVERY_CUH
