#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SUCCESSOR_SHADOWS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SUCCESSOR_SHADOWS_CUH

inline constexpr std::uint32_t kResidentSuccessorShadowCapacity = 4u;

enum class ResidentSuccessorShadowState : std::uint32_t {
  free = 0u,
  live = 1u,
  expired = 2u,
};

struct alignas(8) ResidentSuccessorShadowEntry {
  ResidentRecipeOccurrence occurrence;
  ResidentExecutableMorphologyWork work;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t parent_revision_identity;
  std::uint32_t parent_context_signature;
  std::uint32_t parent_frontier_slot;
  std::uint32_t derivation_index;
  std::uint32_t generation_tick;
  std::uint32_t horizon_tick;
  std::int32_t projected_state_q16;
  std::uint32_t work_units;
  ResidentSuccessorShadowState state;
  std::uint32_t reserved[2];
};
static_assert(std::is_standard_layout_v<ResidentSuccessorShadowEntry> &&
              std::is_trivial_v<ResidentSuccessorShadowEntry> &&
              std::has_unique_object_representations_v<
                  ResidentSuccessorShadowEntry>);

struct alignas(8) ResidentSuccessorShadowFrontier {
  ResidentSuccessorShadowEntry entries[kResidentSuccessorShadowCapacity];
  std::uint32_t live_count;
  std::uint32_t generations;
  std::uint32_t refusals;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentSuccessorShadowFrontier> &&
              std::is_trivial_v<ResidentSuccessorShadowFrontier> &&
              std::has_unique_object_representations_v<
                  ResidentSuccessorShadowFrontier>);

DIRECT_ADULT_HD inline std::uint64_t resident_successor_shadow_identity(
    const ResidentRecipeOccurrence& parent, std::uint32_t generation_tick,
    std::uint32_t horizon_tick) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x7375636373686164ull, parent.occurrence_identity);
  identity = exact_history_fold_word(identity, parent.logical_recipe_id);
  identity = exact_history_fold_word(identity, parent.revision_identity);
  identity = exact_history_fold_word(identity, parent.context_signature);
  identity = exact_history_fold_word(identity, generation_tick);
  identity = exact_history_fold_word(identity, horizon_tick);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint32_t expire_resident_successor_shadows(
    ResidentSuccessorShadowFrontier* frontier, std::uint32_t current_tick) {
  if (frontier == nullptr) return 0u;
  std::uint32_t live = 0u;
  for (std::uint32_t i = 0u; i < kResidentSuccessorShadowCapacity; ++i) {
    auto& entry = frontier->entries[i];
    if (entry.state == ResidentSuccessorShadowState::live &&
        entry.horizon_tick < current_tick) {
      entry.state = ResidentSuccessorShadowState::expired;
      entry.occurrence.state = kResidentRecipeOccurrenceSettled;
    }
    live += entry.state == ResidentSuccessorShadowState::live ? 1u : 0u;
  }
  frontier->live_count = live;
  return live;
}

// Generate one touched, one-step endogenous successor from an exact live
// actual Occurrence. The prospective state may guide later resident work, but
// it owns no external participation, evidence, eligibility, or credit.
DIRECT_ADULT_HD inline bool generate_resident_successor_shadow(
    const DirectBrain& brain, const ResidentActualFrontier& actual_frontier,
    std::uint32_t parent_frontier_slot, std::uint32_t current_tick,
    std::uint32_t horizon_ticks,
    ResidentSuccessorShadowFrontier* shadow_frontier) {
  using namespace direct_network;
  if (shadow_frontier == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      parent_frontier_slot >= kResidentActualFrontierCapacity ||
      horizon_ticks == 0u || current_tick > 0xffffffffu - horizon_ticks) {
    if (shadow_frontier != nullptr) ++shadow_frontier->refusals;
    return false;
  }
  const ResidentActualFrontierEntry& parent =
      actual_frontier.entries[parent_frontier_slot];
  const ResidentRecipeOccurrence& source = parent.occurrence;
  if (parent.state != ResidentActualFrontierState::live ||
      source.state != kResidentRecipeOccurrenceLive ||
      source.lineage_kind != ResidentOccurrenceLineageKind::actual ||
      source.authority == DirectParticipationAuthority::none ||
      source.occurrence_identity == 0u || source.logical_recipe_id == 0u ||
      source.revision_identity == 0u || source.participation_identity == 0u ||
      source.source_identity == 0u || source.source_incarnation == 0u ||
      current_tick < source.timestamp || current_tick > source.expiry_tick ||
      parent.derivation_index >=
          brain.postbirth_constructor->derivation_count ||
      brain.postbirth_constructor->derivation_count >
          kResidentPostbirthRecipeReserve) {
    ++shadow_frontier->refusals;
    return false;
  }
  const ResidentRecipeDerivation& derivation =
      brain.postbirth_derivations[parent.derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
      derivation.logical_recipe_id != source.logical_recipe_id ||
      derivation.revision_identity != source.revision_identity ||
      derivation.port_count != source.binding_count ||
      parent.work.identity == 0u ||
      parent.work.identity != resident_executable_work_identity(parent.work) ||
      parent.work.occurrence_identity != source.occurrence_identity ||
      parent.work.revision_identity != source.revision_identity) {
    ++shadow_frontier->refusals;
    return false;
  }
  const ResidentRecipeCell& recipe = brain.recipe_cells[derivation.recipe_cell];
  if (recipe.logical_recipe_id != source.logical_recipe_id ||
      recipe.revision_identity != source.revision_identity) {
    ++shadow_frontier->refusals;
    return false;
  }
  expire_resident_successor_shadows(shadow_frontier, current_tick);
  for (std::uint32_t i = 0u; i < kResidentSuccessorShadowCapacity; ++i)
    if (shadow_frontier->entries[i].state ==
            ResidentSuccessorShadowState::live &&
        shadow_frontier->entries[i].parent_occurrence_identity ==
            source.occurrence_identity) {
      ++shadow_frontier->refusals;
      return false;
    }
  std::uint32_t slot = kInvalidIndex;
  for (std::uint32_t i = 0u; i < kResidentSuccessorShadowCapacity; ++i)
    if (slot == kInvalidIndex &&
        shadow_frontier->entries[i].state !=
            ResidentSuccessorShadowState::live)
      slot = i;
  if (slot == kInvalidIndex) {
    ++shadow_frontier->refusals;
    return false;
  }
  std::uint32_t variables[kResidentDerivationWidth]{};
  for (std::uint32_t i = 0u; i < source.binding_count; ++i) {
    if (source.bindings[i].formal_port_index != i ||
        source.bindings[i].variable_identity == 0u) {
      ++shadow_frontier->refusals;
      return false;
    }
    variables[i] = source.bindings[i].variable_identity;
  }
  const std::uint32_t horizon_tick = current_tick + horizon_ticks;
  const std::uint64_t shadow_identity = resident_successor_shadow_identity(
      source, current_tick, horizon_tick);
  ResidentSuccessorShadowEntry candidate{};
  if (!bind_resident_recipe_occurrence(
          recipe, derivation, variables, source.binding_count, shadow_identity,
          shadow_identity, source.occurrence_identity,
          source.source_incarnation, ResidentOccurrenceLineageKind::endogenous,
          DirectParticipationAuthority::none, source.context_signature,
          current_tick, horizon_tick, 0, &candidate.occurrence) ||
      !lower_resident_executable_morphology(
          brain, derivation, candidate.occurrence, false, &candidate.work) ||
      !execute_resident_executable_morphology(
          brain, derivation, candidate.occurrence, candidate.work,
          parent.output_q16, &candidate.projected_state_q16,
          &candidate.work_units) ||
      !apply_resident_occurrence_bound_activation(
          &candidate.occurrence, shadow_identity, source.context_signature,
          variables, source.binding_count, current_tick,
          candidate.projected_state_q16)) {
    ++shadow_frontier->refusals;
    return false;
  }
  candidate.parent_occurrence_identity = source.occurrence_identity;
  candidate.parent_revision_identity = source.revision_identity;
  candidate.parent_context_signature = source.context_signature;
  candidate.parent_frontier_slot = parent_frontier_slot;
  candidate.derivation_index = parent.derivation_index;
  candidate.generation_tick = current_tick;
  candidate.horizon_tick = horizon_tick;
  candidate.state = ResidentSuccessorShadowState::live;
  shadow_frontier->entries[slot] = candidate;
  ++shadow_frontier->live_count;
  ++shadow_frontier->generations;
  return true;
}

#endif
