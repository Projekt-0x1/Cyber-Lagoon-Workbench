#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_EXECUTABLE_MORPHOLOGY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_EXECUTABLE_MORPHOLOGY_CUH
// Mid-include splice for substrate::direct_adult_core (see direct_adult_core.cuh).
// Owns lower/execute of Occurrence+derivation into executable morphology work.

DIRECT_ADULT_HD inline bool execute_resident_self_compiled_macro(
    const DirectBrain& brain, const direct_network::ResidentRecipeDerivation& d,
    std::int32_t input_q16, std::int32_t* output_q16, std::uint32_t* work_units) {
  if (output_q16 == nullptr || work_units == nullptr || d.relation_count != 2u ||
      d.parameter_count != 4u || d.parameters_q16[3] != 2 ||
      brain.development == nullptr || brain.postbirth_constructor == nullptr ||
      d.recipe_cell >= brain.development->recipe_cell_count ||
      brain.recipe_cells[d.recipe_cell].logical_recipe_id != d.logical_recipe_id ||
      brain.recipe_cells[d.recipe_cell].revision_identity != d.revision_identity) return false;
  const std::uint32_t a = d.relations[0], b = d.relations[1];
  if (a >= brain.route_capacity || b >= brain.route_capacity ||
      brain.route_incarnations == nullptr) return false;
  const DirectRoute& first = brain.routes[a]; const DirectRoute& second = brain.routes[b];
  if (!direct_network::route_is_active(first) || !direct_network::route_is_active(second) ||
      first.source != d.ports[0].node || first.target != second.source ||
      second.target != d.ports[1].node || brain.route_incarnations[a] != d.route_incarnations[0] ||
      brain.route_incarnations[b] != d.route_incarnations[1] ||
      first.conductance_q16 != d.parameters_q16[1] || second.conductance_q16 != d.parameters_q16[2]) return false;
  if (d.parameters_q16[0] !=
      static_cast<std::int32_t>((static_cast<std::int64_t>(d.parameters_q16[1]) *
                                 d.parameters_q16[2]) >>
                                16)) {
    // Drifted composed parameter leaves the compiled macro's guard domain:
    // refuse the closed form and deopt to the interpreted walk over the
    // validated route conductances, counted, never silently divergent.
    ++brain.postbirth_constructor->macro_param_refusals;
    std::int64_t value =
        (static_cast<std::int64_t>(input_q16) * first.conductance_q16) >> 16;
    value = (value * second.conductance_q16) >> 16;
    if (value < -0x80000000ll || value > 0x7fffffffll) return false;
    *output_q16 = static_cast<std::int32_t>(value);
    *work_units = 2u;
    return true;
  }
  *output_q16 = static_cast<std::int32_t>((static_cast<std::int64_t>(input_q16) * d.parameters_q16[0]) >> 16);
  *work_units = 1u; return true;
}
enum class ResidentExecutableWorkClass : std::uint16_t {
  generic_route_micrograph = 1u,
  compiled_macro = 2u,
};
struct alignas(8) ResidentExecutableMorphologyWork {
  std::uint64_t identity, logical_recipe_id, revision_identity;
  std::uint64_t occurrence_identity, participation_identity;
  std::uint64_t route_incarnations[2];
  std::uint32_t route_indices[2];
  std::int32_t conductances_q16[2];
  std::uint32_t variable_identities[direct_network::kResidentDerivationWidth];
  std::uint32_t context_signature;
  ResidentExecutableWorkClass work_class;
  std::uint16_t binding_count, route_count, reserved;
  std::uint32_t reserved2;
};
static_assert(std::is_standard_layout_v<ResidentExecutableMorphologyWork> &&
              std::is_trivial_v<ResidentExecutableMorphologyWork> &&
              std::has_unique_object_representations_v<
                  ResidentExecutableMorphologyWork>);
DIRECT_ADULT_HD inline bool resident_executable_derivation_equal(
    const direct_network::ResidentRecipeDerivation& left,
    const direct_network::ResidentRecipeDerivation& right) {
  const auto* a = reinterpret_cast<const unsigned char*>(&left);
  const auto* b = reinterpret_cast<const unsigned char*>(&right);
  for (std::uint32_t i = 0u; i < sizeof(left); ++i)
    if (a[i] != b[i]) return false;
  return true;
}
DIRECT_ADULT_HD inline std::uint64_t resident_executable_work_identity(
    const ResidentExecutableMorphologyWork& work) {
  std::uint64_t identity = direct_network::exact_history_fold_word(
      0x657865636d6f7270ull, work.logical_recipe_id);
  identity = direct_network::exact_history_fold_word(identity, work.revision_identity);
  identity = direct_network::exact_history_fold_word(identity, work.occurrence_identity);
  identity = direct_network::exact_history_fold_word(identity, work.participation_identity);
  identity = direct_network::exact_history_fold_word(identity, work.context_signature);
  identity = direct_network::exact_history_fold_word(
      identity, static_cast<std::uint16_t>(work.work_class));
  identity = direct_network::exact_history_fold_word(identity, work.binding_count);
  identity = direct_network::exact_history_fold_word(identity, work.route_count);
  for (std::uint32_t i = 0u;
       i < work.binding_count && i < direct_network::kResidentDerivationWidth; ++i)
    identity = direct_network::exact_history_fold_word(
        identity, work.variable_identities[i]);
  for (std::uint32_t i = 0u; i < work.route_count && i < 2u; ++i) {
    identity = direct_network::exact_history_fold_word(identity, work.route_indices[i]);
    identity = direct_network::exact_history_fold_word(identity, work.route_incarnations[i]);
    identity = direct_network::exact_history_fold_word(
        identity, static_cast<std::uint32_t>(work.conductances_q16[i]));
  }
  return identity == 0u ? 1u : identity;
}
DIRECT_ADULT_HD inline bool lower_resident_executable_morphology(
    const DirectBrain& brain,
    const direct_network::ResidentRecipeDerivation& derivation,
    const ResidentRecipeOccurrence& occurrence, bool generic_fallback,
    ResidentExecutableMorphologyWork* out) {
  using namespace direct_network;
  if (out == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr || brain.postbirth_derivations == nullptr ||
      derivation.recipe_cell >= brain.development->recipe_cell_count ||
      derivation.port_count != 2u ||
      occurrence.state != kResidentRecipeOccurrenceLive ||
      occurrence.logical_recipe_id != derivation.logical_recipe_id ||
      occurrence.revision_identity != derivation.revision_identity ||
      occurrence.binding_count != derivation.port_count ||
      occurrence.occurrence_identity == 0u || occurrence.participation_identity == 0u)
    return false;
  const ResidentRecipeCell& recipe = brain.recipe_cells[derivation.recipe_cell];
  if (recipe.logical_recipe_id != derivation.logical_recipe_id ||
      recipe.revision_identity != derivation.revision_identity)
    return false;
  bool exact_derivation = false;
  for (std::uint32_t i = 0u;
       i < brain.postbirth_constructor->derivation_count; ++i)
    exact_derivation |= resident_executable_derivation_equal(
        derivation, brain.postbirth_derivations[i]);
  if (!exact_derivation) return false;
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    if (occurrence.bindings[i].formal_port_index != i ||
        occurrence.bindings[i].variable_identity == 0u)
      return false;
  }
  ResidentExecutableMorphologyWork work{};
  work.logical_recipe_id = derivation.logical_recipe_id;
  work.revision_identity = derivation.revision_identity;
  work.occurrence_identity = occurrence.occurrence_identity;
  work.participation_identity = occurrence.participation_identity;
  work.context_signature = occurrence.context_signature;
  work.binding_count = occurrence.binding_count;
  for (std::uint32_t i = 0u; i < occurrence.binding_count; ++i)
    work.variable_identities[i] = occurrence.bindings[i].variable_identity;
  const bool trigger = derivation.relation_count == 1u && derivation.parameter_count == 1u &&
      derivation.relations[0] == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger);
  const bool route_gain = derivation.relation_count == 1u && derivation.parameter_count == 1u &&
      derivation.relations[0] == static_cast<std::uint32_t>(direct_network::ResidentRecipeRelation::route_gain);
  const bool condensed = (trigger || route_gain) &&
      (derivation.condensation_flags & kResidentDerivationCondensedNetwork) != 0u;
  if (condensed) {
    ResidentNetworkCondensationEvidence evidence{};
    if (!rematerialize_resident_condensation(derivation, &evidence)) return false;
    work.work_class = ResidentExecutableWorkClass::compiled_macro;
    work.route_count = 0u;
  } else if (trigger || route_gain) {
    if (derivation.route_index >= brain.route_capacity || brain.route_incarnations == nullptr ||
        !route_is_active(brain.routes[derivation.route_index]) ||
        brain.routes[derivation.route_index].source != derivation.ports[0].node ||
        brain.routes[derivation.route_index].target != derivation.ports[1].node ||
        brain.route_incarnations[derivation.route_index] != derivation.route_incarnations[0])
      return false;
    work.work_class = ResidentExecutableWorkClass::generic_route_micrograph;
    work.route_count = 1u;
    work.route_indices[0] = derivation.route_index;
    work.route_incarnations[0] = derivation.route_incarnations[0];
    work.conductances_q16[0] = brain.routes[derivation.route_index].conductance_q16;
  } else {
    const std::uint32_t first_index = derivation.relations[0];
    const std::uint32_t second_index = derivation.relations[1];
    if (derivation.relation_count != 2u || derivation.parameter_count != 4u ||
        derivation.parameters_q16[3] != 2 || first_index >= brain.route_capacity ||
        second_index >= brain.route_capacity || brain.route_incarnations == nullptr)
      return false;
    const DirectRoute& first = brain.routes[first_index];
    const DirectRoute& second = brain.routes[second_index];
    if (!route_is_active(first) || !route_is_active(second) ||
        first.source != derivation.ports[0].node || first.target != second.source ||
        second.target != derivation.ports[1].node ||
        brain.route_incarnations[first_index] != derivation.route_incarnations[0] ||
        brain.route_incarnations[second_index] != derivation.route_incarnations[1])
      return false;
    work.work_class = generic_fallback ||
            first.conductance_q16 != derivation.parameters_q16[1] ||
            second.conductance_q16 != derivation.parameters_q16[2]
        ? ResidentExecutableWorkClass::generic_route_micrograph
        : ResidentExecutableWorkClass::compiled_macro;
    work.route_count = 2u;
    work.route_indices[0] = first_index; work.route_indices[1] = second_index;
    work.route_incarnations[0] = derivation.route_incarnations[0];
    work.route_incarnations[1] = derivation.route_incarnations[1];
    work.conductances_q16[0] = first.conductance_q16;
    work.conductances_q16[1] = second.conductance_q16;
  }
  work.identity = resident_executable_work_identity(work);
  *out = work;
  return true;
}
DIRECT_ADULT_HD inline bool execute_resident_executable_morphology(
    const DirectBrain& brain,
    const direct_network::ResidentRecipeDerivation& derivation,
    const ResidentRecipeOccurrence& occurrence,
    const ResidentExecutableMorphologyWork& work, std::int32_t input_q16,
    std::int32_t* output_q16, std::uint32_t* work_units) {
  if (output_q16 == nullptr || work_units == nullptr ||
      work.identity == 0u || work.identity != resident_executable_work_identity(work))
    return false;
  ResidentExecutableMorphologyWork exact{};
  const bool generic =
      work.work_class == ResidentExecutableWorkClass::generic_route_micrograph;
  if ((!generic && work.work_class != ResidentExecutableWorkClass::compiled_macro) ||
      !lower_resident_executable_morphology(
          brain, derivation, occurrence, generic, &exact) ||
      exact.identity != work.identity)
    return false;
  if (derivation.relation_count == 1u && derivation.parameter_count == 1u) {
    const bool condensed =
        (derivation.condensation_flags &
         direct_network::kResidentDerivationCondensedNetwork) != 0u;
    if (condensed) {
      // Any condensed answer, single-hop or deoptimized, is eligible only
      // from exact current source revision/generation/incarnation snapshots
      // behind a live witness; a stale or corrupt witness fails closed.
      direct_network::ResidentNetworkCondensationEvidence evidence{};
      std::uint64_t logical = 0u, revision = 0u;
      if (!direct_network::rematerialize_resident_condensation(
              derivation, &evidence) ||
          !direct_network::replay_resident_condensation_witness(
              brain.recipe_cells, brain.development->recipe_cell_count,
              brain.postbirth_derivations,
              brain.postbirth_constructor->derivation_count, evidence,
              &logical, &revision))
        return false;
      bool route_parameter_drift = false;
      for (std::uint32_t i = 0u;
           i < direct_network::kResidentCondensationSourceCount; ++i) {
        const auto& source = evidence.sources[i];
        if (source.route_index >= brain.route_capacity ||
            !direct_network::route_is_active(brain.routes[source.route_index]) ||
            brain.route_incarnations[source.route_index] !=
                source.route_incarnation)
          return false;
        if (source.relation == static_cast<std::uint32_t>(
                                   direct_network::ResidentRecipeRelation::route_gain) &&
            brain.routes[source.route_index].conductance_q16 != source.parameter_q16)
          route_parameter_drift = true;
      }
      if (route_parameter_drift) {
        std::int64_t value = input_q16;
        for (std::uint32_t i = 0u;
             i < direct_network::kResidentCondensationSourceCount; ++i) {
          const auto& source = evidence.sources[i];
          if (source.relation != static_cast<std::uint32_t>(
                                     direct_network::ResidentRecipeRelation::route_gain))
            return false;
          value = (value * brain.routes[source.route_index].conductance_q16) >> 16;
        }
        if (value < -0x80000000ll || value > 0x7fffffffll) return false;
        ++brain.postbirth_constructor->macro_param_refusals;
        *output_q16 = static_cast<std::int32_t>(value);
        *work_units = direct_network::kResidentCondensationSourceCount;
        return true;
      }
      if (input_q16 >= 0 &&
          (static_cast<std::uint32_t>(input_q16) <
               derivation.condensation_guard_min_q16 ||
           static_cast<std::uint32_t>(input_q16) >
               derivation.condensation_guard_max_q16))
        return direct_network::execute_deoptimized_resident_condensation(
            derivation, input_q16, brain.postbirth_constructor, output_q16,
            work_units);
      if (derivation.relations[0] == static_cast<std::uint32_t>(
                                         direct_network::ResidentRecipeRelation::route_gain)) {
        std::int32_t value = input_q16;
        for (std::uint32_t i = 0u;
             i < direct_network::kResidentCondensationSourceCount; ++i)
          if (!direct_network::evaluate_resident_recipe_boundary_q16(
                  evidence.sources[i].relation, evidence.sources[i].parameter_q16,
                  value, &value))
            return false;
        *output_q16 = value;
        *work_units = 1u;
        return true;
      }
    }
    if (!direct_network::evaluate_resident_recipe_boundary_q16(
            derivation.relations[0], derivation.parameters_q16[0], input_q16,
            output_q16))
      return false;
    *work_units = 1u;
    return true;
  }
  if (!generic)
    return execute_resident_self_compiled_macro(
        brain, derivation, input_q16, output_q16, work_units);
  std::int64_t value = input_q16;
  for (std::uint32_t i = 0u; i < exact.route_count; ++i)
    value = (value * exact.conductances_q16[i]) >> 16;
  if (value < -0x80000000ll || value > 0x7fffffffll) return false;
  *output_q16 = static_cast<std::int32_t>(value);
  *work_units = exact.route_count;
  return true;
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_EXECUTABLE_MORPHOLOGY_CUH
