#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_WHITEBOX_OBSERVE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_WHITEBOX_OBSERVE_CUH
// Mid-include splice. Mixed-rank N -> B_N -> exact affine witness.
// Does not replace observe_resident_network_condensation (trigger/route_gain).

DIRECT_ADULT_HD inline std::uint32_t resident_network_occ_index(
    const ResidentRecipeOccurrence* occurrences, std::uint32_t n,
    std::uint64_t occurrence_identity) {
  for (std::uint32_t i = 0; i < n; ++i)
    if (occurrences[i].occurrence_identity == occurrence_identity) return i;
  return n;
}

DIRECT_ADULT_HD inline bool resident_network_boundary_relation_equal(
    const ResidentRelationalNetworkClosure& left,
    const ResidentRelationalNetworkClosure& right) {
  if (left.boundary_count != right.boundary_count || left.boundary_count < 2u ||
      left.occurrence_count != right.occurrence_count)
    return false;
  for (std::uint16_t i = 0; i < left.boundary_count; ++i) {
    bool found = false;
    for (std::uint16_t j = 0; j < right.boundary_count; ++j)
      if (left.boundary[i].variable_identity == right.boundary[j].variable_identity &&
          left.boundary[i].direction == right.boundary[j].direction)
        found = true;
    if (!found) return false;
  }
  return true;
}

DIRECT_ADULT_HD inline bool fold_resident_mixed_rank_affine(
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* occurrences,
    const ResidentOccurrenceCoupling* edges, std::uint32_t occurrence_count,
    std::uint32_t coupling_count, const ResidentRelationalNetworkClosure& closure,
    direct_network::DirectWhiteboxCondensationV1* witness) {
  using namespace direct_network;
  if (derivations == nullptr || occurrences == nullptr || edges == nullptr ||
      witness == nullptr)
    return false;
  std::uint64_t occ = 0;
  std::uint32_t inputs = 0;
  for (std::uint16_t i = 0; i < closure.boundary_count; ++i)
    if (closure.boundary[i].direction == ResidentRecipePortDirection::input) {
      occ = closure.boundary[i].occurrence_identity;
      ++inputs;
    }
  if (inputs != 1u) return false;
  std::int32_t acc[2]{};
  bool acc_set = false;
  std::uint32_t parameterized = 0, visited = 0;
  std::uint8_t used[kResidentRelationalNetworkMaxOccurrences]{};
  *witness = {};
  while (visited < occurrence_count) {
    const std::uint32_t idx = resident_network_occ_index(occurrences, occurrence_count, occ);
    if (idx >= occurrence_count || used[idx] != 0u) return false;
    used[idx] = 1u;
    ++visited;
    const auto& der = derivations[idx];
    const bool identity = der.parameter_count == 0u && der.relation_count == 0u;
    const bool linear = der.parameter_count == 2u && der.relation_count == 1u &&
                        direct_relation_algebra_word_is_family(
                            der.relations[0], DirectRelationAlgebraFamilyV1::linear);
    if (!identity && !linear) return false;
    if (linear) {
      ++parameterized;
      if (parameterized > 3u) return false;
      if (!acc_set) {
        acc[0] = der.parameters_q16[0];
        acc[1] = der.parameters_q16[1];
        acc_set = true;
      } else {
        DirectWhiteboxReductionSourceV1 source{};
        source.kind = DirectWhiteboxReductionKindV1::affine_composition;
        source.coefficient_count = 4u;
        source.coefficients_q16[0] = acc[0];
        source.coefficients_q16[1] = acc[1];
        source.coefficients_q16[2] = der.parameters_q16[0];
        source.coefficients_q16[3] = der.parameters_q16[1];
        source.source_identity = direct_whitebox_source_identity(source);
        if (!direct_whitebox_reduce_exact(source, witness) || witness->result_count != 2u)
          return false;
        acc[0] = witness->result_q16[0];
        acc[1] = witness->result_q16[1];
      }
    }
    std::uint64_t next = 0;
    std::uint32_t outs = 0;
    for (std::uint32_t e = 0; e < coupling_count; ++e)
      if (edges[e].source_occurrence_identity == occ) {
        next = edges[e].target_occurrence_identity;
        ++outs;
      }
    if (outs > 1u) return false;
    if (outs == 0u) break;
    occ = next;
  }
  return visited == occurrence_count && parameterized >= 2u &&
         witness->witness_identity != 0u && direct_whitebox_condensation_valid(*witness);
}

DIRECT_ADULT_HD inline bool observe_resident_mixed_rank_whitebox(
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* first_occ,
    const ResidentRecipeOccurrence* second_occ, std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* first_edges,
    const ResidentOccurrenceCoupling* second_edges, std::uint32_t coupling_count,
    direct_network::DirectWhiteboxCondensationV1* witness) {
  if (recipes == nullptr || derivations == nullptr || first_occ == nullptr ||
      second_occ == nullptr || first_edges == nullptr || second_edges == nullptr ||
      witness == nullptr || occurrence_count < 2u)
    return false;
  for (std::uint32_t i = 0; i < occurrence_count; ++i) {
    if (first_occ[i].lineage_kind != ResidentOccurrenceLineageKind::actual ||
        second_occ[i].lineage_kind != ResidentOccurrenceLineageKind::actual)
      return false;
    for (std::uint32_t j = 0; j < occurrence_count; ++j)
      if (first_occ[i].occurrence_identity == second_occ[j].occurrence_identity ||
          first_occ[i].participation_identity == second_occ[j].participation_identity)
        return false;
  }
  ResidentRelationalNetworkClosure first{};
  ResidentRelationalNetworkClosure second{};
  if (!bind_resident_relational_network_closure(recipes, derivations, first_occ,
                                                occurrence_count, first_edges,
                                                coupling_count, &first) ||
      !bind_resident_relational_network_closure(recipes, derivations, second_occ,
                                                occurrence_count, second_edges,
                                                coupling_count, &second) ||
      first.actual_count != occurrence_count ||
      second.actual_count != occurrence_count ||
      !resident_network_boundary_relation_equal(first, second))
    return false;
  direct_network::DirectWhiteboxCondensationV1 again{};
  return fold_resident_mixed_rank_affine(derivations, first_occ, first_edges,
                                         occurrence_count, coupling_count, first,
                                         witness) &&
         fold_resident_mixed_rank_affine(derivations, second_occ, second_edges,
                                         occurrence_count, coupling_count, second,
                                         &again) &&
         again.witness_identity == witness->witness_identity &&
         again.result_q16[0] == witness->result_q16[0] &&
         again.result_q16[1] == witness->result_q16[1];
}

DIRECT_ADULT_HD inline bool collect_resident_mixed_rank_sources(
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* occurrences,
    const ResidentOccurrenceCoupling* edges, std::uint32_t occurrence_count,
    std::uint32_t coupling_count, const ResidentRelationalNetworkClosure& closure,
    std::uint32_t* first_idx, std::uint32_t* second_idx) {
  using namespace direct_network;
  if (derivations == nullptr || occurrences == nullptr || edges == nullptr ||
      first_idx == nullptr || second_idx == nullptr)
    return false;
  std::uint64_t occ = 0;
  std::uint32_t inputs = 0;
  for (std::uint16_t i = 0; i < closure.boundary_count; ++i)
    if (closure.boundary[i].direction == ResidentRecipePortDirection::input) {
      occ = closure.boundary[i].occurrence_identity;
      ++inputs;
    }
  if (inputs != 1u) return false;
  std::uint32_t found[2] = {occurrence_count, occurrence_count};
  std::uint32_t parameterized = 0, visited = 0;
  std::uint8_t used[kResidentRelationalNetworkMaxOccurrences]{};
  while (visited < occurrence_count) {
    const std::uint32_t idx = resident_network_occ_index(occurrences, occurrence_count, occ);
    if (idx >= occurrence_count || used[idx] != 0u) return false;
    used[idx] = 1u;
    ++visited;
    const auto& der = derivations[idx];
    if (der.parameter_count == 2u && der.relation_count == 1u &&
        direct_relation_algebra_word_is_family(der.relations[0],
                                              DirectRelationAlgebraFamilyV1::linear)) {
      if (parameterized >= 2u) return false;
      found[parameterized++] = idx;
    } else if (!(der.parameter_count == 0u && der.relation_count == 0u))
      return false;
    std::uint64_t next = 0;
    std::uint32_t outs = 0;
    for (std::uint32_t e = 0; e < coupling_count; ++e)
      if (edges[e].source_occurrence_identity == occ) {
        next = edges[e].target_occurrence_identity;
        ++outs;
      }
    if (outs > 1u) return false;
    if (outs == 0u) break;
    occ = next;
  }
  if (visited != occurrence_count || parameterized != 2u) return false;
  *first_idx = found[0];
  *second_idx = found[1];
  return true;
}

DIRECT_ADULT_HD inline std::uint64_t resident_mixed_rank_observation(
    const ResidentRecipeOccurrence* occurrences, std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* edges, std::uint32_t coupling_count,
    std::uint64_t whitebox_witness) {
  using namespace direct_network;
  std::uint64_t observation = exact_history_fold_word(0x6e65746f62736572ull, whitebox_witness);
  for (std::uint32_t i = 0; i < occurrence_count; ++i) {
    observation = exact_history_fold_word(observation, occurrences[i].occurrence_identity);
    observation = exact_history_fold_word(observation, occurrences[i].participation_identity);
    observation = exact_history_fold_word(observation, occurrences[i].source_identity);
    observation = exact_history_fold_word(observation, occurrences[i].source_incarnation);
  }
  for (std::uint32_t e = 0; e < coupling_count; ++e)
    observation = exact_history_fold_word(observation, edges[e].variable_identity);
  return observation == 0u ? 1u : observation;
}

DIRECT_ADULT_HD inline bool observe_resident_mixed_rank_evidence(
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* first_occ,
    const ResidentRecipeOccurrence* second_occ, std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* first_edges,
    const ResidentOccurrenceCoupling* second_edges, std::uint32_t coupling_count,
    direct_network::DirectWhiteboxCondensationV1* witness,
    direct_network::ResidentNetworkCondensationEvidence* evidence) {
  using namespace direct_network;
  if (evidence == nullptr ||
      !observe_resident_mixed_rank_whitebox(recipes, derivations, first_occ, second_occ,
                                           occurrence_count, first_edges, second_edges,
                                           coupling_count, witness))
    return false;
  ResidentRelationalNetworkClosure first{};
  if (!bind_resident_relational_network_closure(recipes, derivations, first_occ,
                                                occurrence_count, first_edges,
                                                coupling_count, &first))
    return false;
  std::uint32_t a = 0, b = 0;
  if (!collect_resident_mixed_rank_sources(derivations, first_occ, first_edges,
                                           occurrence_count, coupling_count, first, &a,
                                           &b))
    return false;
  const std::int32_t guard_a = first_occ[a].activation_q16;
  const std::int32_t guard_b = second_occ[a].activation_q16;
  if (guard_a < 0 || guard_b < 0 || guard_a == guard_b) return false;
  ResidentNetworkCondensationEvidence candidate{};
  candidate.source_count = kResidentCondensationSourceCount;
  candidate.recurring_observations = 2u;
  candidate.maximum_error_q16 = 0;
  candidate.guard_min_q16 =
      static_cast<std::uint32_t>(guard_a < guard_b ? guard_a : guard_b);
  candidate.guard_max_q16 =
      static_cast<std::uint32_t>(guard_a < guard_b ? guard_b : guard_a);
  const std::uint32_t idxs[2] = {a, b};
  for (std::uint32_t i = 0; i < 2u; ++i) {
    const std::uint32_t idx = idxs[i];
    candidate.sources[i] = ResidentCondensationSourceSnapshot{
        recipes[idx].logical_recipe_id, recipes[idx].revision_identity,
        derivations[idx].generation, derivations[idx].route_incarnations[0],
        first_occ[idx].source_identity, derivations[idx].recipe_cell,
        direct_relation_algebra_resident_word(DirectRelationAlgebraFamilyV1::linear),
        derivations[idx].route_index,
        derivations[idx].parameters_q16[0], derivations[idx].ports[0],
        derivations[idx].ports[1]};
  }
  candidate.observation_identities[0] = resident_mixed_rank_observation(
      first_occ, occurrence_count, first_edges, coupling_count, witness->witness_identity);
  candidate.observation_identities[1] = resident_mixed_rank_observation(
      second_occ, occurrence_count, second_edges, coupling_count,
      witness->witness_identity);
  candidate.observation_variable_identities[0] = first_edges[0].variable_identity;
  candidate.observation_variable_identities[1] = second_edges[0].variable_identity;
  candidate.observation_source_incarnations[0][0] = first_occ[a].source_incarnation;
  candidate.observation_source_incarnations[0][1] = first_occ[b].source_incarnation;
  candidate.observation_source_incarnations[1][0] = second_occ[a].source_incarnation;
  candidate.observation_source_incarnations[1][1] = second_occ[b].source_incarnation;
  if (candidate.observation_identities[0] == candidate.observation_identities[1] ||
      candidate.observation_source_incarnations[0][0] ==
          candidate.observation_source_incarnations[1][0])
    return false;
  candidate.refinement_logical_recipe_id = recipes[a].logical_recipe_id;
  candidate.refinement_revision_identity = recipes[a].revision_identity;
  candidate.refinement_recipe_cell = derivations[a].recipe_cell;
  candidate.boundary_identity = resident_condensation_boundary_identity(candidate);
  candidate.proof_identity = resident_condensation_proof_identity(candidate);
  candidate.witness_identity = resident_network_condensation_witness(candidate);
  if (candidate.witness_identity == 0u || candidate.boundary_identity == 0u ||
      candidate.proof_identity == 0u)
    return false;
  *evidence = candidate;
  return true;
}

DIRECT_ADULT_HD inline bool resident_mixed_rank_source_matches(
    const direct_network::ResidentCondensationSourceSnapshot& source,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count) {
  using namespace direct_network;
  if (cells == nullptr || derivations == nullptr || source.recipe_cell >= cell_count ||
      source.logical_recipe_id == 0u || source.revision_identity == 0u ||
      source.source_identity == 0u ||
      cells[source.recipe_cell].logical_recipe_id != source.logical_recipe_id ||
      cells[source.recipe_cell].revision_identity != source.revision_identity)
    return false;
  for (std::uint32_t i = 0; i < derivation_count; ++i) {
    const auto& derivation = derivations[i];
    if (derivation.logical_recipe_id != source.logical_recipe_id ||
        derivation.recipe_cell != source.recipe_cell)
      continue;
    return derivation.generation == source.generation &&
           derivation.recipe_cell == source.recipe_cell &&
           derivation.route_index == source.route_index &&
           derivation.route_incarnations[0] == source.route_incarnation &&
           derivation.relation_count == 1u && derivation.parameter_count == 2u &&
           derivation.port_count == 2u &&
           source.relation ==
               direct_relation_algebra_resident_word(DirectRelationAlgebraFamilyV1::linear) &&
           direct_relation_algebra_word_is_family(derivation.relations[0],
                                                 DirectRelationAlgebraFamilyV1::linear) &&
           derivation.parameters_q16[0] == source.parameter_q16 &&
           derivation.ports[0].node == source.input_port.node &&
           derivation.ports[0].domain == source.input_port.domain &&
           derivation.ports[0].direction == source.input_port.direction &&
           derivation.ports[0].arity == source.input_port.arity &&
           derivation.ports[1].node == source.output_port.node &&
           derivation.ports[1].domain == source.output_port.domain &&
           derivation.ports[1].direction == source.output_port.direction &&
           derivation.ports[1].arity == source.output_port.arity;
  }
  return false;
}

DIRECT_ADULT_HD inline bool resident_mixed_rank_witness_is_source_fold(
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const direct_network::ResidentNetworkCondensationEvidence& evidence,
    const direct_network::DirectWhiteboxCondensationV1& witness) {
  using namespace direct_network;
  const ResidentRecipeDerivation* first = nullptr;
  const ResidentRecipeDerivation* second = nullptr;
  for (std::uint32_t i = 0; i < derivation_count; ++i) {
    if (first == nullptr &&
        derivations[i].logical_recipe_id == evidence.sources[0].logical_recipe_id &&
        derivations[i].recipe_cell == evidence.sources[0].recipe_cell)
      first = &derivations[i];
    if (second == nullptr &&
        derivations[i].logical_recipe_id == evidence.sources[1].logical_recipe_id &&
        derivations[i].recipe_cell == evidence.sources[1].recipe_cell)
      second = &derivations[i];
  }
  if (first == nullptr || second == nullptr || first->parameter_count != 2u ||
      second->parameter_count != 2u)
    return false;
  DirectWhiteboxReductionSourceV1 source{};
  source.kind = DirectWhiteboxReductionKindV1::affine_composition;
  source.coefficient_count = 4u;
  source.coefficients_q16[0] = first->parameters_q16[0];
  source.coefficients_q16[1] = first->parameters_q16[1];
  source.coefficients_q16[2] = second->parameters_q16[0];
  source.coefficients_q16[3] = second->parameters_q16[1];
  source.source_identity = direct_whitebox_source_identity(source);
  DirectWhiteboxCondensationV1 folded{};
  return direct_whitebox_reduce_exact(source, &folded) && folded.result_count == 2u &&
         resident_whitebox_recipe_logical_identity(folded) ==
             resident_whitebox_recipe_logical_identity(witness);
}

DIRECT_ADULT_HD inline bool replay_resident_mixed_rank_evidence(
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const direct_network::ResidentNetworkCondensationEvidence& evidence,
    const direct_network::DirectWhiteboxCondensationV1& witness,
    std::uint64_t* logical_recipe_id, std::uint64_t* revision_identity) {
  using namespace direct_network;
  if (cells == nullptr || derivations == nullptr || logical_recipe_id == nullptr ||
      revision_identity == nullptr || !direct_whitebox_condensation_valid(witness) ||
      evidence.source_count != kResidentCondensationSourceCount ||
      evidence.recurring_observations < 2u ||
      evidence.guard_min_q16 >= evidence.guard_max_q16 ||
      evidence.maximum_error_q16 != 0 ||
      evidence.boundary_identity != resident_condensation_boundary_identity(evidence) ||
      evidence.proof_identity != resident_condensation_proof_identity(evidence) ||
      evidence.witness_identity != resident_network_condensation_witness(evidence) ||
      evidence.refinement_logical_recipe_id == 0u ||
      evidence.refinement_revision_identity == 0u ||
      evidence.refinement_recipe_cell >= cell_count ||
      cells[evidence.refinement_recipe_cell].logical_recipe_id !=
          evidence.refinement_logical_recipe_id ||
      cells[evidence.refinement_recipe_cell].revision_identity !=
          evidence.refinement_revision_identity)
    return false;
  for (std::uint32_t i = 0; i < kResidentCondensationSourceCount; ++i)
    if (evidence.observation_identities[i] == 0u ||
        evidence.observation_variable_identities[i] == 0u ||
        evidence.observation_source_incarnations[i][0] == 0u ||
        evidence.observation_source_incarnations[i][1] == 0u ||
        !resident_mixed_rank_source_matches(evidence.sources[i], cells, cell_count,
                                            derivations, derivation_count))
      return false;
  if (evidence.observation_source_incarnations[0][0] ==
          evidence.observation_source_incarnations[1][0] ||
      evidence.sources[0].recipe_cell == evidence.sources[1].recipe_cell ||
      evidence.sources[0].relation != evidence.sources[1].relation ||
      !resident_recipe_ports_compatible(evidence.sources[0].output_port,
                                        evidence.sources[1].input_port) ||
      !resident_mixed_rank_witness_is_source_fold(derivations, derivation_count, evidence,
                                                 witness))
    return false;
  const std::uint64_t logical = resident_whitebox_recipe_logical_identity(witness);
  const std::int64_t support =
      cells[evidence.sources[0].recipe_cell].support_q16 <
              cells[evidence.sources[1].recipe_cell].support_q16
          ? cells[evidence.sources[0].recipe_cell].support_q16
          : cells[evidence.sources[1].recipe_cell].support_q16;
  *logical_recipe_id = logical;
  *revision_identity = resident_recipe_revision_identity(
      logical, 0u, 1u, witness.witness_identity, support, 0);
  return logical != 0u && *revision_identity != 0u;
}

DIRECT_ADULT_HD inline bool replay_resident_network_candidate(
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const direct_network::ResidentNetworkCondensationEvidence& evidence,
    const direct_network::DirectWhiteboxCondensationV1* whitebox,
    std::uint64_t* logical_recipe_id, std::uint64_t* revision_identity) {
  using namespace direct_network;
  if (evidence.source_count != kResidentCondensationSourceCount) return false;
  DirectRelationAlgebraFamilyV1 family_a{}, family_b{};
  const bool tagged_a =
      direct_relation_algebra_resident_word_parse(evidence.sources[0].relation, &family_a,
                                                 nullptr);
  const bool tagged_b =
      direct_relation_algebra_resident_word_parse(evidence.sources[1].relation, &family_b,
                                                 nullptr);
  if (tagged_a && tagged_b) {
    if (whitebox == nullptr || family_a != family_b) return false;
    return replay_resident_mixed_rank_evidence(cells, cell_count, derivations,
                                              derivation_count, evidence, *whitebox,
                                              logical_recipe_id, revision_identity);
  }
  const std::uint32_t rel = evidence.sources[0].relation;
  if (rel != evidence.sources[1].relation ||
      (rel != static_cast<std::uint32_t>(ResidentRecipeRelation::trigger) &&
       rel != static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain)))
    return false;
  return replay_resident_condensation_witness(cells, cell_count, derivations,
                                             derivation_count, evidence, logical_recipe_id,
                                             revision_identity);
}

#endif
