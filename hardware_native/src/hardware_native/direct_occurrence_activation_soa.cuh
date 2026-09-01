#ifndef HARDWARE_NATIVE_DIRECT_OCCURRENCE_ACTIVATION_SOA_CUH
#define HARDWARE_NATIVE_DIRECT_OCCURRENCE_ACTIVATION_SOA_CUH

// Included by direct_adult_actual_frontier.cuh inside direct_adult_core after
// the canonical frontier types exist. This cache is replaceable execution
// infrastructure; it never owns Occurrence identity or exact history. The
// canonical AoS frontier stays the source of truth: every consumer validates
// currency against the control cursor and falls back to canonical reads when
// the plane is stale, poisoned, or absent.
struct alignas(16) ResidentOccurrenceActivationSoa {
  std::int32_t activation_q16[kResidentActualFrontierCapacity]{};
  std::uint64_t occurrence_identity[kResidentActualFrontierCapacity]{};
  std::uint64_t revision_identity[kResidentActualFrontierCapacity]{};
  std::uint64_t participation_identity[kResidentActualFrontierCapacity]{};
  std::uint32_t source_slot[kResidentActualFrontierCapacity]{};
  std::uint32_t count = 0u;
  std::uint32_t source_live_count = 0u;
  std::uint64_t mutation_watermark = 0u;
  std::uint64_t published_epoch = 0u;
};

// Device-owned invalidation cursor. Production mutators of the canonical live
// set (expiry, admission, condensation settlement) bump frontier_mutations;
// refreshes publish refresh_epoch. A plane is current only when both match its
// latched copies and the canonical live count still agrees.
struct alignas(8) ResidentActivationSoaControl {
  std::uint64_t frontier_mutations = 0u;
  std::uint64_t refresh_epoch = 0u;
};

struct alignas(16) ResidentActivationSoaPlane {
  ResidentOccurrenceActivationSoa soa{};
  ResidentActivationSoaControl control{};
};

DIRECT_ADULT_HD inline bool project_resident_actual_frontier_activation_soa(
    const ResidentActualFrontier& frontier, ResidentOccurrenceActivationSoa* out) {
  if (out == nullptr || frontier.live_count > kResidentActualFrontierCapacity)
    return false;
  ResidentOccurrenceActivationSoa candidate{};
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const auto& entry = frontier.entries[i];
    if (entry.state != ResidentActualFrontierState::live)
      continue;
    const auto& occurrence = entry.occurrence;
    if (candidate.count >= kResidentActualFrontierCapacity ||
        occurrence.state != kResidentRecipeOccurrenceLive || occurrence.occurrence_identity == 0u ||
        occurrence.revision_identity == 0u || occurrence.participation_identity == 0u)
      return false;
    const std::uint32_t slot = candidate.count++;
    candidate.activation_q16[slot] = occurrence.activation_q16;
    candidate.occurrence_identity[slot] = occurrence.occurrence_identity;
    candidate.revision_identity[slot] = occurrence.revision_identity;
    candidate.participation_identity[slot] = occurrence.participation_identity;
    candidate.source_slot[slot] = i;
  }
  if (candidate.count != frontier.live_count)
    return false;
  candidate.source_live_count = frontier.live_count;
  *out = candidate;
  return true;
}

// Refresh the derived activation plane from canonical storage. Failure poisons
// the plane: published_epoch returns to zero so no consumer can treat a partial
// projection as current.
DIRECT_ADULT_HD inline void refresh_resident_actual_frontier_activation_soa(
    const ResidentActualFrontier& frontier,
    ResidentActivationSoaControl* control,
    ResidentOccurrenceActivationSoa* soa) {
  if (control == nullptr || soa == nullptr)
    return;
  ++control->refresh_epoch;
  if (!project_resident_actual_frontier_activation_soa(frontier, soa)) {
    soa->count = 0u;
    soa->source_live_count = 0u;
    soa->published_epoch = 0u;
    soa->mutation_watermark = control->frontier_mutations;
    return;
  }
  soa->mutation_watermark = control->frontier_mutations;
  soa->published_epoch = control->refresh_epoch;
}

DIRECT_ADULT_HD inline bool resident_activation_soa_current(
    const ResidentActualFrontier& frontier,
    const ResidentActivationSoaControl& control,
    const ResidentOccurrenceActivationSoa& soa) {
  return soa.published_epoch != 0u && soa.published_epoch == control.refresh_epoch &&
         soa.mutation_watermark == control.frontier_mutations &&
         soa.count == soa.source_live_count &&
         soa.source_live_count == frontier.live_count &&
         frontier.live_count <= kResidentActualFrontierCapacity;
}

#endif
