#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_CONDENSATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_CONDENSATION_CUH
// Mid-include splice for substrate::direct_adult_core (see direct_adult_core.cuh).
// Owns condensation observe + compact dynamic-closure reuse of that quotient.

// Two independently bound recurrences nominate a composition; production
// derives and probes the boundary relation itself, so the caller supplies no
// expected answer or host-authored meta-recipe body.
DIRECT_ADULT_HD inline bool observe_resident_network_condensation(
    const direct_network::ResidentRecipeCell (&recipes)[2],
    const direct_network::ResidentRecipeDerivation (&derivations)[2],
    const ResidentRecipeOccurrence (&left_occurrences)[2],
    const ResidentRecipeOccurrence (&right_occurrences)[2],
    const ResidentOccurrenceCoupling (&couplings)[2],
    direct_network::ResidentNetworkCondensationEvidence* evidence) {
  using namespace direct_network;
  if (evidence == nullptr) return false;
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    if (recipes[i].logical_recipe_id == 0u || recipes[i].revision_identity == 0u ||
        derivations[i].logical_recipe_id != recipes[i].logical_recipe_id ||
        derivations[i].revision_identity != recipes[i].revision_identity ||
        derivations[i].port_count != 2u || derivations[i].relation_count != 1u ||
        derivations[i].parameter_count != 1u ||
        (derivations[i].relations[0] !=
             static_cast<std::uint32_t>(ResidentRecipeRelation::trigger) &&
         derivations[i].relations[0] !=
             static_cast<std::uint32_t>(direct_network::ResidentRecipeRelation::route_gain)))
      return false;
  }
  if (derivations[0].relations[0] != derivations[1].relations[0]) return false;
  if (!resident_recipe_ports_compatible(
          derivations[0].ports[1], derivations[1].ports[0]))
    return false;
  for (std::uint32_t repetition = 0u; repetition < 2u; ++repetition) {
    const auto& left = left_occurrences[repetition];
    const auto& right = right_occurrences[repetition];
    const auto& coupling = couplings[repetition];
    if (left.state != kResidentRecipeOccurrenceLive ||
        right.state != kResidentRecipeOccurrenceLive ||
        left.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        right.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        left.authority == DirectParticipationAuthority::none ||
        right.authority == DirectParticipationAuthority::none ||
        coupling.source_occurrence_identity != left.occurrence_identity ||
        coupling.target_occurrence_identity != right.occurrence_identity ||
        coupling.source_revision_identity != recipes[0].revision_identity ||
        coupling.target_revision_identity != recipes[1].revision_identity ||
        left.source_identity == 0u || right.source_identity == 0u ||
        left.source_identity != left_occurrences[0].source_identity ||
        right.source_identity != right_occurrences[0].source_identity ||
        left.route_incarnation != derivations[0].route_incarnations[0] ||
        right.route_incarnation != derivations[1].route_incarnations[0] ||
        coupling.source_incarnation != left.source_incarnation ||
        coupling.target_incarnation != right.source_incarnation ||
        coupling.source_port_index != 1u || coupling.target_port_index != 0u ||
        coupling.variable_identity == 0u)
      return false;
  }
  if (left_occurrences[0].occurrence_identity ==
          left_occurrences[1].occurrence_identity ||
      right_occurrences[0].occurrence_identity ==
          right_occurrences[1].occurrence_identity ||
      left_occurrences[0].participation_identity ==
          left_occurrences[1].participation_identity)
    return false;
  const std::int32_t first = left_occurrences[0].activation_q16;
  const std::int32_t second = left_occurrences[1].activation_q16;
  if (first < 0 || second < 0 || first == second) return false;
  ResidentNetworkCondensationEvidence candidate{};
  candidate.source_count = kResidentCondensationSourceCount;
  candidate.recurring_observations = 2u;
  candidate.guard_min_q16 = static_cast<std::uint32_t>(first < second ? first : second);
  candidate.guard_max_q16 = static_cast<std::uint32_t>(first < second ? second : first);
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    candidate.sources[i] = ResidentCondensationSourceSnapshot{
        recipes[i].logical_recipe_id, recipes[i].revision_identity,
        derivations[i].generation, derivations[i].route_incarnations[0],
        i == 0u ? left_occurrences[0].source_identity
                : right_occurrences[0].source_identity,
        derivations[i].recipe_cell, derivations[i].relations[0],
        derivations[i].route_index, derivations[i].parameters_q16[0],
        derivations[i].ports[0], derivations[i].ports[1]};
  }
  for (std::uint32_t repetition = 0u; repetition < 2u; ++repetition) {
    std::uint64_t observation = direct_network::exact_history_fold_word(
        0x6e65746f62736572ull,
        left_occurrences[repetition].occurrence_identity);
    observation = direct_network::exact_history_fold_word(
        observation, right_occurrences[repetition].occurrence_identity);
    observation = direct_network::exact_history_fold_word(
        observation, left_occurrences[repetition].participation_identity);
    observation = direct_network::exact_history_fold_word(
        observation, right_occurrences[repetition].participation_identity);
    observation = direct_network::exact_history_fold_word(
        observation, couplings[repetition].variable_identity);
    observation = direct_network::exact_history_fold_word(
        observation, left_occurrences[repetition].source_identity);
    observation = direct_network::exact_history_fold_word(
        observation, right_occurrences[repetition].source_identity);
    observation = direct_network::exact_history_fold_word(
        observation, left_occurrences[repetition].route_incarnation);
    observation = direct_network::exact_history_fold_word(
        observation, right_occurrences[repetition].route_incarnation);
    observation = direct_network::exact_history_fold_word(
        observation, couplings[repetition].source_incarnation);
    observation = direct_network::exact_history_fold_word(
        observation, couplings[repetition].target_incarnation);
    candidate.observation_identities[repetition] =
        observation == 0u ? 1u : observation;
    candidate.observation_source_incarnations[repetition][0] =
        left_occurrences[repetition].source_incarnation;
    candidate.observation_source_incarnations[repetition][1] =
        right_occurrences[repetition].source_incarnation;
    candidate.observation_variable_identities[repetition] =
        couplings[repetition].variable_identity;
  }
  const std::uint32_t condensed_relation = derivations[0].relations[0];
  const std::int64_t condensed_parameter =
      condensed_relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger)
          ? static_cast<std::int64_t>(derivations[0].parameters_q16[0]) +
                derivations[1].parameters_q16[0]
          : (static_cast<std::int64_t>(derivations[0].parameters_q16[0]) *
                derivations[1].parameters_q16[0]) >> 16;
  if (condensed_parameter < -0x80000000ll || condensed_parameter > 0x7fffffffll)
    return false;
  const std::int32_t probes[3] = {
      static_cast<std::int32_t>(candidate.guard_min_q16),
      static_cast<std::int32_t>(candidate.guard_min_q16 +
          (candidate.guard_max_q16 - candidate.guard_min_q16) / 2u),
      static_cast<std::int32_t>(candidate.guard_max_q16)};
  std::int32_t maximum_error = 0;
  for (std::uint32_t i = 0u; i < 3u; ++i) {
    std::int32_t intermediate = 0, composed = 0, condensed = 0;
    if (!evaluate_resident_recipe_boundary_q16(
            derivations[0].relations[0], derivations[0].parameters_q16[0],
            probes[i], &intermediate) ||
        !evaluate_resident_recipe_boundary_q16(
            derivations[1].relations[0], derivations[1].parameters_q16[0],
            intermediate, &composed))
      return false;
    if (condensed_relation ==
        static_cast<std::uint32_t>(direct_network::ResidentRecipeRelation::route_gain)) {
      condensed = probes[i];
      for (std::uint32_t source = 0u; source < 2u; ++source)
        if (!evaluate_resident_recipe_boundary_q16(
                derivations[source].relations[0],
                derivations[source].parameters_q16[0], condensed, &condensed))
          return false;
    } else if (!evaluate_resident_recipe_boundary_q16(
                   condensed_relation,
                   static_cast<std::int32_t>(condensed_parameter), probes[i],
                   &condensed))
      return false;
    const std::int32_t error = composed >= condensed
        ? composed - condensed : condensed - composed;
    if (error > maximum_error) maximum_error = error;
  }
  candidate.maximum_error_q16 = maximum_error;
  candidate.boundary_identity = resident_condensation_boundary_identity(candidate);
  candidate.refinement_logical_recipe_id = recipes[0].logical_recipe_id;
  candidate.refinement_revision_identity = recipes[0].revision_identity;
  candidate.refinement_recipe_cell = derivations[0].recipe_cell;
  candidate.proof_identity = resident_condensation_proof_identity(candidate);
  candidate.witness_identity = resident_network_condensation_witness(candidate);
  if (maximum_error != 0 || candidate.witness_identity == 0u) return false;
  *evidence = candidate;
  return true;
}
struct alignas(8) ResidentExecutionFacts {
  std::uint64_t witness_identity;
  std::int32_t output_q16;
  std::uint32_t work_units, detailed_support_live, compact_witness_live;
  std::uint32_t guard_open, plasticity_responsive, compact_reuse, reserved;
};
static_assert(std::is_standard_layout_v<ResidentExecutionFacts> &&
              std::is_trivial_v<ResidentExecutionFacts> &&
              std::has_unique_object_representations_v<ResidentExecutionFacts>);

// Regional/Latent/River are observer names for these resident facts, never
// stored classes. Detailed source relations remain the lawful fallback; an
// exact condensation makes their quotient reusable, and current lived support
// alone opens or closes its recorded guard.
DIRECT_ADULT_HD inline bool execute_resident_dynamic_closure(
    const direct_network::ResidentRecipeCell (&recipes)[2],
    const direct_network::ResidentRecipeDerivation (&sources)[2],
    const ResidentRecipeOccurrence (&support)[2],
    const direct_network::ResidentRecipeDerivation* condensed,
    std::int32_t input_q16, ResidentExecutionFacts* out) {
  using namespace direct_network;
  if (out == nullptr) return false;
  std::int32_t intermediate = 0, detailed = 0;
  for (std::uint32_t i = 0u; i < 2u; ++i)
    if (recipes[i].logical_recipe_id == 0u ||
        sources[i].logical_recipe_id != recipes[i].logical_recipe_id ||
        sources[i].revision_identity != recipes[i].revision_identity ||
        sources[i].port_count != 2u || sources[i].relation_count != 1u ||
        sources[i].parameter_count != 1u ||
        sources[i].relations[0] !=
            static_cast<std::uint32_t>(ResidentRecipeRelation::trigger))
      return false;
  if (!evaluate_resident_recipe_boundary_q16(
          sources[0].relations[0], sources[0].parameters_q16[0], input_q16,
          &intermediate) ||
      !evaluate_resident_recipe_boundary_q16(
          sources[1].relations[0], sources[1].parameters_q16[0], intermediate,
          &detailed))
    return false;
  ResidentExecutionFacts facts{};
  facts.output_q16 = detailed;
  facts.work_units = 2u;
  facts.detailed_support_live = 1u;
  facts.plasticity_responsive = 1u;
  if (condensed == nullptr ||
      (condensed->condensation_flags & kResidentDerivationCondensedNetwork) == 0u)
    { *out = facts; return true; }
  ResidentNetworkCondensationEvidence evidence{};
  if (!rematerialize_resident_condensation(*condensed, &evidence))
    { *out = facts; return true; }
  bool current = condensed->logical_recipe_id != 0u &&
      condensed->revision_identity != 0u &&
      condensed->condensation_source_count == 2u;
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    const auto& snapshot = condensed->condensation_sources[i];
    current &= snapshot.logical_recipe_id == recipes[i].logical_recipe_id &&
        snapshot.revision_identity == recipes[i].revision_identity &&
        snapshot.generation == sources[i].generation &&
        snapshot.recipe_cell == sources[i].recipe_cell &&
        snapshot.relation == sources[i].relations[0] &&
        snapshot.parameter_q16 == sources[i].parameters_q16[0] &&
        support[i].state == kResidentRecipeOccurrenceLive &&
        support[i].logical_recipe_id == recipes[i].logical_recipe_id &&
        support[i].revision_identity == recipes[i].revision_identity &&
        support[i].authority != DirectParticipationAuthority::none &&
        support[i].activation_q16 > 0;
  }
  if (!current) { *out = facts; return true; }
  std::int32_t compact = 0;
  if (!evaluate_resident_recipe_boundary_q16(
          condensed->relations[0], condensed->parameters_q16[0], input_q16,
          &compact) || compact != detailed)
    return false;
  const std::uint32_t guard_signal = static_cast<std::uint32_t>(support[0].activation_q16);
  facts.witness_identity = condensed->witness_identity;
  facts.output_q16 = compact;
  facts.compact_witness_live = 1u;
  facts.compact_reuse = 1u;
  facts.plasticity_responsive = 0u;
  facts.guard_open = guard_signal >= condensed->condensation_guard_min_q16 &&
      guard_signal <= condensed->condensation_guard_max_q16;
  facts.work_units = facts.guard_open != 0u ? 1u : 2u;
  *out = facts;
  return true;
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_CONDENSATION_CUH
