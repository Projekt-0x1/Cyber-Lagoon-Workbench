#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTION_NETWORK_UNFOLDING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTION_NETWORK_UNFOLDING_CUH

// Mid-include splice for direct_adult_actual_frontier.cuh. Owns the hot
// action-side unfolding of one current relational Network component.

#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
// Freeze the exact current closure and its compact resident recruitment on the
// action. The closure is transiently unfolded from current Occurrences; only the
// reusable recruitment/credit mathematics persists.
__device__ inline bool resident_action_recruited_network_identity(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    DirectActionParticipationLink* links, std::uint32_t participant_offset,
    std::uint32_t* participant_count, std::uint32_t participant_capacity,
    const DirectParticipationDescriptor* current_contributions,
    std::uint32_t current_contribution_count,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations,
    std::uint32_t current_tick,
    std::uint64_t* network_identity, std::uint64_t* recruitment_identity,
    std::int64_t* eligibility_signed_q16, std::uint64_t* eligibility_l1_q16) {
  // Canonical motor publication has one physical owner. Keep the bounded
  // unfolded set in block scratch instead of multiplying a large local frame
  // across every CUDA thread in the motor/persistent kernels.
  if (blockIdx.x != 0u || threadIdx.x != 0u) return false;
  if (network_identity == nullptr || recruitment_identity == nullptr ||
      eligibility_signed_q16 == nullptr || eligibility_l1_q16 == nullptr)
    return false;
  *network_identity = 0u;
  *recruitment_identity = 0u;
  *eligibility_signed_q16 = 0;
  *eligibility_l1_q16 = 0u;
  if (frontier == nullptr || links == nullptr || brain.development == nullptr ||
      participant_count == nullptr || *participant_count == 0u ||
      *participant_count > participant_capacity ||
      participant_capacity > kMaxActionParticipationLinks ||
      (current_contribution_count > 0u && current_contributions == nullptr) ||
      eligibility_table == nullptr || eligibility_record_generations == nullptr)
    return false;
  std::uint64_t occurrence_ids[kMaxActionParticipationLinks]{};
  std::uint32_t exact_count = 0u;
  for (std::uint32_t p = 0u; p < *participant_count; ++p) {
    const auto& link = links[participant_offset + p];
    if (link.occurrence_identity == 0u) continue;
    bool duplicate = false;
    for (std::uint32_t i = 0u; i < exact_count; ++i)
      duplicate |= occurrence_ids[i] == link.occurrence_identity;
    if (duplicate) continue;
    bool current = false;
    for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
      const auto& entry = frontier->entries[slot];
      current |= entry.state == ResidentActualFrontierState::live &&
          entry.occurrence.state == kResidentRecipeOccurrenceLive &&
          entry.occurrence.expiry_tick >= current_tick &&
          entry.occurrence.occurrence_identity == link.occurrence_identity &&
          entry.occurrence.logical_recipe_id == link.logical_recipe_id &&
          entry.occurrence.revision_identity == link.revision_identity &&
          entry.occurrence.participation_identity == link.participation_identity &&
          entry.occurrence.authority == link.authority;
    }
    if (!current) return false;
    occurrence_ids[exact_count++] = link.occurrence_identity;
  }
  if (exact_count == 0u) return true;

  __shared__ direct_network::ResidentRecipeCell
      recipes[kResidentRelationalNetworkMaxOccurrences];
  __shared__ direct_network::ResidentRecipeDerivation
      derivations[kResidentRelationalNetworkMaxOccurrences];
  __shared__ ResidentRecipeOccurrence
      occurrences[kResidentRelationalNetworkMaxOccurrences];
  __shared__ ResidentOccurrenceCoupling
      couplings[kResidentRelationalNetworkMaxCouplings];
  __shared__ ResidentRelationalNetworkClosure closure;
  __shared__ DirectParticipationDescriptor member_descriptors[
      kResidentRelationalNetworkMaxOccurrences];
  __shared__ bool member_descriptor_ready[
      kResidentRelationalNetworkMaxOccurrences];
  std::uint32_t occurrence_count = 0u, coupling_count = 0u;
  for (std::uint32_t i = 0u;
       i < kResidentRelationalNetworkMaxOccurrences; ++i) {
    member_descriptors[i] = DirectParticipationDescriptor{};
    member_descriptor_ready[i] = false;
  }
  if (!collect_resident_actual_frontier_relational_network(
          brain, *frontier, recipes, derivations, occurrences, couplings,
          &occurrence_count, &coupling_count))
    return false;

  // The phase staging is the sparse touched frontier. Reuse its exact
  // eligibility references instead of scanning resident morphology or the
  // whole live eligibility table to unfold current Network members.
  for (std::uint32_t occurrence = 0u; occurrence < occurrence_count;
       ++occurrence) {
    const auto& resident = occurrences[occurrence];
    const auto& derivation = derivations[occurrence];
    DirectParticipationDescriptor selected{};
    std::uint32_t matches = 0u;
    for (std::uint32_t candidate = 0u;
         candidate < current_contribution_count; ++candidate) {
      const auto& value = current_contributions[candidate];
      if (value.ticket_id != resident.participation_identity ||
          value.route_index != derivation.route_index ||
          value.route_incarnation != resident.route_incarnation ||
          value.claim_incarnation != resident.source_incarnation ||
          value.authority != resident.authority ||
          value.contribution_kind != DirectContributionKind::sparse_route ||
          value.expiry_tick < current_tick ||
          value.frozen_eligibility_q16 <= 0)
        continue;
      if (value.eligibility_slot == kInvalidIndex ||
          value.eligibility_slot >= kMaxLiveEligibilityRecords ||
          value.eligibility_generation == 0u ||
          atomicAdd(const_cast<std::uint32_t*>(
                        eligibility_record_generations +
                        value.eligibility_slot),
                    0u) != value.eligibility_generation)
        continue;
      const EligibilityRecord record =
          eligibility_table[value.eligibility_slot];
      __threadfence();
      if (atomicAdd(const_cast<std::uint32_t*>(
                        eligibility_record_generations +
                        value.eligibility_slot),
                    0u) != value.eligibility_generation ||
          record.live == 0u || record.expiry_tick < current_tick ||
          record.lineage_expiry_tick < current_tick ||
          record.ticket_id != value.ticket_id ||
          record.source_node != value.source_node ||
          record.target_node != value.target_node ||
          record.route_index != value.route_index ||
          record.context_signature != value.context_signature ||
          record.route_incarnation != value.route_incarnation ||
          record.claim_incarnation != value.claim_incarnation ||
          record.authority != value.authority ||
          record.authority_incarnation != value.authority_incarnation ||
          record.parent_eligibility_ref != value.parent_eligibility_ref ||
          record.ancestry_depth != value.ancestry_depth ||
          record.eligibility_q16 != value.frozen_eligibility_q16)
        continue;
      selected = value;
      ++matches;
    }
    if (matches > 1u) return false;
    if (matches == 0u && resident.eligibility_ref != 0u) {
      const std::uint32_t slot =
          static_cast<std::uint32_t>(resident.eligibility_ref) - 1u;
      const std::uint32_t generation =
          static_cast<std::uint32_t>(resident.eligibility_ref >> 32u);
      if (slot < kMaxLiveEligibilityRecords && generation != 0u &&
          atomicAdd(const_cast<std::uint32_t*>(
                        eligibility_record_generations + slot),
                    0u) == generation) {
        const EligibilityRecord record = eligibility_table[slot];
        __threadfence();
        if (atomicAdd(const_cast<std::uint32_t*>(
                          eligibility_record_generations + slot),
                      0u) == generation &&
            record.live != 0u && record.expiry_tick >= current_tick &&
            record.lineage_expiry_tick >= current_tick &&
            record.ticket_id == resident.participation_identity &&
            record.route_index == derivation.route_index &&
            record.route_incarnation == resident.route_incarnation &&
            record.claim_incarnation == resident.source_incarnation &&
            resident_occurrence_accepts_causal_authority(
                resident.authority, record.authority) &&
            record.eligibility_q16 > 0) {
          selected.ticket_id = record.ticket_id;
          selected.source_node = record.source_node;
          selected.target_node = record.target_node;
          selected.route_index = record.route_index;
          selected.context_signature = record.context_signature;
          selected.expiry_tick = record.expiry_tick;
          selected.claim_incarnation = record.claim_incarnation;
          selected.route_incarnation = record.route_incarnation;
          selected.authority = record.authority;
          selected.authority_incarnation = record.authority_incarnation;
          selected.contribution_kind = DirectContributionKind::sparse_route;
          selected.frozen_eligibility_q16 = record.eligibility_q16;
          selected.eligibility_slot = slot;
          selected.eligibility_generation = generation;
          selected.parent_eligibility_ref = record.parent_eligibility_ref;
          selected.lineage_expiry_tick = record.lineage_expiry_tick;
          selected.ancestry_depth = record.ancestry_depth;
          matches = 1u;
        }
      }
    }
    if (matches == 1u) {
      occurrences[occurrence].eligibility_q16 =
          selected.frozen_eligibility_q16;
      member_descriptors[occurrence] = selected;
      member_descriptor_ready[occurrence] = true;
    }
  }

  // Unfold only the connected component that contains the exact motor seed.
  // This is an in-place compaction of the already sparse current frontier: it
  // neither scans dormant morphology nor parks four full closures in each CUDA
  // block merely because unrelated Networks are coactive.
  std::uint8_t component = 0u;
  for (std::uint32_t seed = 0u; seed < exact_count; ++seed) {
    std::uint32_t match = occurrence_count;
    for (std::uint32_t i = 0u; i < occurrence_count; ++i)
      if (occurrences[i].occurrence_identity == occurrence_ids[seed]) {
        if (match != occurrence_count) return false;
        match = i;
      }
    if (match == occurrence_count) return true;
    if (seed == 0u)
      component = static_cast<std::uint8_t>(1u << match);
  }
  for (std::uint32_t pass = 0u; pass < occurrence_count; ++pass)
    for (std::uint32_t c = 0u; c < coupling_count; ++c) {
      std::uint32_t source = occurrence_count, target = occurrence_count;
      for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
        if (occurrences[i].occurrence_identity ==
            couplings[c].source_occurrence_identity)
          source = i;
        if (occurrences[i].occurrence_identity ==
            couplings[c].target_occurrence_identity)
          target = i;
      }
      if (source == occurrence_count || target == occurrence_count)
        return false;
      if ((component & static_cast<std::uint8_t>(1u << source)) != 0u)
        component = static_cast<std::uint8_t>(
            component | static_cast<std::uint8_t>(1u << target));
      if ((component & static_cast<std::uint8_t>(1u << target)) != 0u)
        component = static_cast<std::uint8_t>(
            component | static_cast<std::uint8_t>(1u << source));
    }
  for (std::uint32_t seed = 0u; seed < exact_count; ++seed) {
    bool contained = false;
    for (std::uint32_t i = 0u; i < occurrence_count; ++i)
      contained |= occurrences[i].occurrence_identity == occurrence_ids[seed] &&
          (component & static_cast<std::uint8_t>(1u << i)) != 0u;
    // The base action may legitimately combine several causal Occurrences
    // without those Occurrences forming one variable-bound relational
    // Network. Preserve that already exact action; only a uniquely connected
    // component may add Network identity or recruit extra participants.
    if (!contained) return true;
  }

  std::uint64_t component_ids[kResidentRelationalNetworkMaxOccurrences]{};
  std::uint32_t component_count = 0u;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    if ((component & static_cast<std::uint8_t>(1u << i)) == 0u) continue;
    component_ids[component_count] = occurrences[i].occurrence_identity;
    recipes[component_count] = recipes[i];
    derivations[component_count] = derivations[i];
    occurrences[component_count] = occurrences[i];
    member_descriptors[component_count] = member_descriptors[i];
    member_descriptor_ready[component_count] = member_descriptor_ready[i];
    ++component_count;
  }
  std::uint32_t component_couplings = 0u;
  for (std::uint32_t c = 0u; c < coupling_count; ++c) {
    bool source = false, target = false;
    for (std::uint32_t i = 0u; i < component_count; ++i) {
      source |= component_ids[i] == couplings[c].source_occurrence_identity;
      target |= component_ids[i] == couplings[c].target_occurrence_identity;
    }
    if (source && target) couplings[component_couplings++] = couplings[c];
  }
  occurrence_count = component_count;
  coupling_count = component_couplings;
  if (occurrence_count < kResidentRelationalNetworkMinOccurrences)
    return true;

  // A coactive closure is not yet an action Network unless every member has
  // exact current action participation. In particular, a first developmental
  // contact can coexist with another live Occurrence while only the motor seed
  // participates. That valid base action must remain publishable, and the
  // incomplete coalition must not mint persistent recruitment morphology.
  for (std::uint32_t member = 0u; member < occurrence_count; ++member)
    if (!member_descriptor_ready[member]) return true;

  if (!bind_resident_relational_network_closure(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, &closure) ||
      !recruit_resident_relational_network(
          brain.development, closure, current_tick))
    return false;
  const auto& recruited = brain.development->recruited_networks;
  const std::uint64_t rid =
      resident_relational_network_recruitment_identity(closure);
  std::uint32_t persistent_matches = 0u, persistent_slot = 0u;
  for (std::uint32_t i = 0u; i < recruited.incidence_count; ++i)
    if (recruited.incidences[i].recruitment_identity == rid) {
      persistent_slot = i;
      ++persistent_matches;
    }
  if (rid == 0u || persistent_matches != 1u ||
      recruited.incidences[persistent_slot].credit_q16 < 0)
    return false;

  // A selected Network is itself causal action structure. Freeze every
  // current member as an ordinary exact action participant before public
  // realization; the closure remains transient while recruitment math stays.
  for (std::uint16_t member = 0u; member < closure.occurrence_count; ++member) {
    const std::uint64_t member_identity =
        closure.members[member].occurrence_identity;
    bool represented = false;
    for (std::uint32_t p = 0u; p < *participant_count; ++p)
      represented |= links[participant_offset + p].occurrence_identity ==
          member_identity;
    if (represented) continue;
    std::uint32_t source = occurrence_count;
    for (std::uint32_t i = 0u; i < occurrence_count; ++i)
      if (occurrences[i].occurrence_identity == member_identity) source = i;
    if (source == occurrence_count || !member_descriptor_ready[source] ||
        *participant_count == participant_capacity)
      return false;
    const auto& descriptor = member_descriptors[source];
    DirectActionParticipationLink link{};
    link.participant_ticket_id = descriptor.ticket_id;
    link.source_node = descriptor.source_node;
    link.target_node = descriptor.target_node;
    link.route_index = descriptor.route_index;
    link.route_incarnation = descriptor.route_incarnation;
    link.context_signature = descriptor.context_signature;
    link.expiry_tick = descriptor.expiry_tick;
    link.claim_incarnation = descriptor.claim_incarnation;
    link.authority_incarnation = descriptor.authority_incarnation;
    link.authority = descriptor.authority;
    link.contribution_kind = descriptor.contribution_kind;
    link.frozen_eligibility_q16 = descriptor.frozen_eligibility_q16;
    link.eligibility_slot = descriptor.eligibility_slot;
    link.eligibility_generation = descriptor.eligibility_generation;
    const auto& resident = occurrences[source];
    if (resident.occurrence_identity != member_identity ||
        resident.participation_identity != descriptor.ticket_id ||
        resident.source_incarnation != descriptor.claim_incarnation ||
        resident.route_incarnation != descriptor.route_incarnation ||
        resident.authority != descriptor.authority)
      return false;
    link.logical_recipe_id = resident.logical_recipe_id;
    link.revision_identity = resident.revision_identity;
    link.occurrence_identity = resident.occurrence_identity;
    link.participation_identity = resident.participation_identity;
    link.occurrence_route_incarnation = resident.route_incarnation;
    link.occurrence_context_signature = resident.context_signature;
    std::uint32_t depth_matches = 0u;
    for (std::uint32_t slot = 0u;
         slot < kResidentActualFrontierCapacity; ++slot) {
      const auto& entry = frontier->entries[slot];
      if (entry.state != ResidentActualFrontierState::live ||
          entry.occurrence.occurrence_identity != member_identity)
        continue;
      link.composition_depth = entry.composition_depth;
      ++depth_matches;
    }
    if (depth_matches != 1u) return false;
    links[participant_offset + *participant_count] = link;
    ++*participant_count;
  }
  if (recruited.incidence_count >
      direct_network::kResidentRecruitedNetworkCapacity)
    return false;
  *network_identity = closure.identity;
  *recruitment_identity = rid;
  *eligibility_signed_q16 = closure.eligibility_signed_q16;
  *eligibility_l1_q16 = closure.eligibility_l1_q16;
  return true;
}
#endif

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_ACTION_NETWORK_UNFOLDING_CUH
