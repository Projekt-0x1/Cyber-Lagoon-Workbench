#ifndef HARDWARE_NATIVE_DIRECT_ADULT_INVERSE_TRANSFORMATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_INVERSE_TRANSFORMATION_CUH

#include "direct_adult_core.cuh"
#include "direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_mathematical_relation_solver.cuh"

namespace substrate::direct_adult_core {
inline constexpr std::int64_t kResidentInverseThetaDeltaLimitQ16 = 16ll << 16;
__device__ inline bool resident_inverse_postcredit_output(
    DirectBrain brain, const direct_network::ResidentRecipeDerivation& derivation,
    const ResidentRecipeOccurrence& occurrence, const direct_network::DirectExactHistoryRecord* prior_recipe_records,
    std::uint32_t prior_recipe_record_count, std::int32_t* output_q16) {
  using namespace direct_network;
  if (output_q16 == nullptr || brain.routes == nullptr ||
      occurrence.state != kResidentRecipeOccurrenceLive ||
      occurrence.revision_identity != derivation.revision_identity) return false;
  auto planned_parameter = [&](std::uint32_t route, std::int32_t base,
                               std::int32_t* out) {
    if (out == nullptr || route >= brain.route_capacity ||
        !route_is_active(brain.routes[route])) return false;
    std::int64_t value = base;
    for (std::uint32_t i = 0u; i < prior_recipe_record_count; ++i) {
      const auto& record = prior_recipe_records[i];
      if (record.kind == DirectExactHistoryKind::recipe_commit &&
          record.subject == route) value += record.resource_delta;
    }
    if (value < -0x80000000ll || value > 0x7fffffffll) return false;
    *out = static_cast<std::int32_t>(value); return true;
  };
  if (derivation.relation_count == 1u && derivation.parameter_count == 1u) {
    std::int32_t parameter = derivation.parameters_q16[0];
    if (derivation.relations[0] == static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain) &&
        !planned_parameter(derivation.route_index, parameter, &parameter)) return false;
    return evaluate_resident_recipe_boundary_q16(derivation.relations[0], parameter,
        occurrence.activation_q16, output_q16);
  }
  if (derivation.relation_count != 2u || derivation.parameter_count != 4u ||
      derivation.parameters_q16[3] != 2) return false;
  std::int32_t first = 0, second = 0;
  if (!planned_parameter(derivation.relations[0], derivation.parameters_q16[1], &first) ||
      !planned_parameter(derivation.relations[1], derivation.parameters_q16[2], &second)) return false;
  std::int64_t value = (static_cast<std::int64_t>(occurrence.activation_q16) * first) >> 16;
  value = (value * second) >> 16;
  if (value < -0x80000000ll || value > 0x7fffffffll) return false;
  *output_q16 = static_cast<std::int32_t>(value); return true;
}
__device__ inline bool inverse_action_link_matches_actual(
    const DirectActionParticipationLink& link,
    const ResidentActualFrontierEntry& entry) {
  const auto& occurrence = entry.occurrence;
  return entry.state == ResidentActualFrontierState::live &&
      occurrence.lineage_kind == ResidentOccurrenceLineageKind::actual &&
      occurrence.authority == DirectParticipationAuthority::independent_external &&
      occurrence.occurrence_identity == link.occurrence_identity &&
      occurrence.logical_recipe_id == link.logical_recipe_id &&
      occurrence.revision_identity == link.revision_identity &&
      occurrence.participation_identity == link.participation_identity &&
      occurrence.context_signature == link.occurrence_context_signature &&
      occurrence.route_incarnation == link.occurrence_route_incarnation &&
      occurrence.source_incarnation == link.claim_incarnation &&
      occurrence.authority == link.authority;
}
__device__ inline std::uint64_t resident_inverse_proposal_identity(
    const ResidentActualFrontierEntry& entry, std::uint64_t action_ticket,
    std::uint32_t recipe_cell, std::uint32_t observed_word,
    const direct_network::DirectInverseThetaSolutionV1& solution,
    std::uint64_t predecessor_identity) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x696e767468657461ull, entry.occurrence.occurrence_identity);
  identity = exact_history_fold_word(identity, action_ticket);
  identity = exact_history_fold_word(identity, entry.revision_identity);
  identity = exact_history_fold_word(identity, predecessor_identity);
  identity = exact_history_fold_word(identity, entry.occurrence.source_identity);
  identity = exact_history_fold_word(identity, entry.occurrence.source_incarnation);
  identity = exact_history_fold_word(identity, entry.occurrence.context_signature);
  identity = exact_history_fold_word(identity, entry.occurrence.expiry_tick);
  identity = exact_history_fold_word(identity, recipe_cell);
  identity = exact_history_fold_word(identity, observed_word);
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(solution.theta_q16));
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(solution.observed_residual_q16));
  identity = exact_history_fold_word(identity, static_cast<std::uint32_t>(solution.solver_kind));
  return identity == 0u ? 1u : identity;
}
__device__ inline std::uint32_t plan_resident_inverse_transformation(
    DirectBrain brain, ResidentActualFrontier* frontier,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, std::uint32_t participant_count,
    std::uint64_t action_ticket, Word observed_word, std::uint32_t commit_tick,
    const direct_network::DirectExactHistoryHotPage& history,
    const direct_network::DirectExactHistoryRecord* prior_recipe_records,
    std::uint32_t prior_recipe_record_count,
    direct_network::DirectExactHistoryRecord* out_revision) {
  using namespace direct_network;
  if (frontier == nullptr || action_links == nullptr || out_revision == nullptr ||
      brain.development == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      action_ticket == 0u || action_ticket == kInvalidTicket ||
      participant_count == 0u || participant_count > kMaxActionParticipationLinks ||
      frontier->live_count > kResidentActualFrontierCapacity)
    return 0u;

  DirectPredecessorShadowTrace trace{};
  const bool trace_complete = trace_direct_predecessor_shadows(history, action_ticket, &trace) &&
      trace.complete != 0u && trace.overflow == 0u;
  const bool verified_trace = trace_complete && trace.provenance == DirectSpeculativeProvenance::verified_world_observation &&
      trace.verified_observation_boundaries != 0u && trace.unverified_observation_boundaries == 0u;
  const bool current_occurrence_suffix = trace.overflow == 0u && trace.evidence_boundaries == 0u &&
      trace.unverified_observation_boundaries == 0u && trace.unresolved_identities <= 1u;
  ResidentActualFrontierEntry* match = nullptr;
  std::uint32_t match_slot = kInvalidIndex;
  for (std::uint32_t p = 0u; p < participant_count; ++p) {
    const auto& link = action_links[participant_offset + p];
    if (link.occurrence_identity == 0u) continue;
    for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
      auto& entry = frontier->entries[slot];
      if (!inverse_action_link_matches_actual(link, entry)) continue;
      if (match != nullptr && match != &entry) return 0u;
      match = &entry;
      match_slot = slot;
    }
  }
  if (match == nullptr || !resident_actual_frontier_entry_current(
                              brain, *brain.postbirth_constructor, *match) ||
      match->derivation_index >= brain.postbirth_constructor->derivation_count)
    return 0u;
  if (!verified_trace && !current_occurrence_suffix) return 0u;
  const auto& derivation = brain.postbirth_derivations[match->derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count)
    return 0u;
  if (static_cast<std::int64_t>(observed_word) == match->output_q16) return 0u;
  ResidentRecipeCell predicted = brain.recipe_cells[derivation.recipe_cell];
  for (std::uint32_t i = 0u; i < prior_recipe_record_count; ++i)
    if (prior_recipe_records[i].kind == DirectExactHistoryKind::recipe_revision &&
        prior_recipe_records[i].source == derivation.recipe_cell &&
        !apply_resident_recipe_revision_event(
            &predicted, prior_recipe_records[i], derivation.recipe_cell))
      return 0u;
  if (predicted.edge_offset > brain.development->recipe_edge_count ||
      static_cast<std::uint32_t>(predicted.edge_count) >
          brain.development->recipe_edge_count - predicted.edge_offset)
    return 0u;
  DirectImplicitRelationRowV1 row{};
  row.variable_index = derivation.recipe_cell;
  row.term_offset = predicted.edge_offset;
  row.theta_q16 =
      direct_relation_saturating_add(predicted.support_q16, predicted.credit_q16);
  row.self_coefficient_q16 = kDirectImplicitSelfCoefficientQ16;
  row.term_count = predicted.edge_count;
  row.flags = predicted.flags;
  std::int32_t predicted_output_q16 = 0;
  if (!resident_inverse_postcredit_output(
          brain, derivation, match->occurrence, prior_recipe_records,
          prior_recipe_record_count,
          &predicted_output_q16))
    return 0u;
  const auto solver_kind = static_cast<DirectRelationSolverKind>(
      (match->occurrence.occurrence_identity ^ action_ticket) %
      kDirectRelationSolverKindCount);
  DirectInverseThetaSolutionV1 solution{};
  if (!direct_relation_inverse_theta_q16(
          row, static_cast<std::int64_t>(observed_word), predicted_output_q16,
          solver_kind, kDirectInverseThetaMaxSteps, &solution) ||
      solution.observed_residual_q16 == 0 ||
      solution.observed_residual_q16 > kResidentInverseThetaDeltaLimitQ16 ||
      solution.observed_residual_q16 < -kResidentInverseThetaDeltaLimitQ16)
    return 0u;

  const std::int64_t predicted_theta = row.theta_q16;
  const std::int64_t delta = solution.theta_q16 - predicted_theta;
  if (delta == 0 || delta > kResidentInverseThetaDeltaLimitQ16 ||
      delta < -kResidentInverseThetaDeltaLimitQ16)
    return 0u;

  const std::uint64_t proposal_identity = resident_inverse_proposal_identity(
      *match, action_ticket, derivation.recipe_cell, observed_word, solution,
      match->occurrence.participation_identity);
  auto& proposal = match->inverse;
  if (proposal.state != ResidentInverseTransformationState::free &&
      (proposal.state != ResidentInverseTransformationState::proposed ||
       proposal.identity != proposal_identity))
    return 0u;
  if (!stage_resident_recipe_revision_event(
          out_revision, predicted, derivation.recipe_cell,
          ResidentRecipeRevisionAuthority::experience,
          match->occurrence.occurrence_identity, action_ticket, commit_tick,
          match->occurrence.timestamp, match_slot, match_slot,
          static_cast<std::uint32_t>(solver_kind), delta))
    return 0u;
  proposal = ResidentInverseTransformationProposal{
      proposal_identity, action_ticket, match->occurrence.occurrence_identity,
      match->revision_identity, match->occurrence.participation_identity,
      out_revision->identity, match->occurrence.source_identity,
      match->occurrence.source_incarnation,
      match->occurrence.context_signature, solution.theta_q16,
      solution.observed_residual_q16, derivation.recipe_cell, observed_word,
      commit_tick, static_cast<std::uint32_t>(solver_kind),
      match->occurrence.expiry_tick,
      ResidentInverseTransformationState::proposed};
  return 1u;
}

__device__ inline bool apply_resident_inverse_transformation(
    const direct_network::DirectExactHistoryRecord& revision,
    direct_network::ResidentRecipeCell* recipe_cells,
    std::uint32_t recipe_cell_count, ResidentActualFrontier* frontier) {
  if (recipe_cells == nullptr || frontier == nullptr ||
      revision.kind != direct_network::DirectExactHistoryKind::recipe_revision ||
      revision.source >= recipe_cell_count)
    return false;
  ResidentInverseTransformationProposal* proposal = nullptr;
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
    auto& candidate = frontier->entries[slot].inverse;
    if (candidate.state != ResidentInverseTransformationState::proposed ||
        candidate.proposed_revision_identity != revision.identity)
      continue;
    if (proposal != nullptr) return false;
    proposal = &candidate;
  }
  if (proposal == nullptr || proposal->recipe_cell != revision.source ||
      !direct_network::apply_resident_recipe_revision_event(
          recipe_cells + revision.source, revision, revision.source))
    return false;
  proposal->state = ResidentInverseTransformationState::committed;
  return true;
}
__device__ inline bool apply_resident_inverse_transformation_and_bindings(
    const direct_network::DirectExactHistoryRecord& revision,
    DirectBrain brain, ResidentActualFrontier* frontier) {
  if (brain.development == nullptr || brain.recipe_cells == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      revision.source >= brain.development->recipe_cell_count ||
      revision.resource_delta == 0) return false;
  const std::uint64_t prior_revision =
      brain.recipe_cells[revision.source].revision_identity;
  for (std::uint32_t d = 0u; d < brain.postbirth_constructor->derivation_count; ++d) {
    const auto& derivation = brain.postbirth_derivations[d];
    if (derivation.recipe_cell != revision.source ||
        derivation.revision_identity != prior_revision ||
        derivation.relation_count != 1u || derivation.parameter_count != 1u ||
        derivation.relations[0] != static_cast<std::uint32_t>(
            direct_network::ResidentRecipeRelation::trigger)) continue;
    const std::int64_t revised_parameter =
        static_cast<std::int64_t>(derivation.parameters_q16[0]) + revision.resource_delta;
    if (revised_parameter < -0x80000000ll || revised_parameter > 0x7fffffffll)
      return false;
  }
  if (!apply_resident_inverse_transformation(
          revision, brain.recipe_cells, brain.development->recipe_cell_count,
          frontier))
    return false;
  for (std::uint32_t d = 0u; d < brain.postbirth_constructor->derivation_count; ++d) {
    auto& derivation = brain.postbirth_derivations[d];
    if (derivation.recipe_cell != revision.source ||
        derivation.revision_identity != prior_revision) continue;
    derivation.revision_identity = brain.recipe_cells[revision.source].revision_identity;
    const bool trigger = derivation.relation_count == 1u && derivation.parameter_count == 1u &&
        derivation.relations[0] ==
        static_cast<std::uint32_t>(direct_network::ResidentRecipeRelation::trigger);
    if (trigger) {
      derivation.parameters_q16[0] = static_cast<std::int32_t>(static_cast<std::int64_t>(
          derivation.parameters_q16[0]) + revision.resource_delta);
    } else if (derivation.relation_count == 1u && derivation.parameter_count == 1u &&
        derivation.relations[0] == static_cast<std::uint32_t>(direct_network::ResidentRecipeRelation::route_gain) &&
        derivation.route_index < brain.route_capacity && direct_network::route_is_active(brain.routes[derivation.route_index]) &&
        brain.route_incarnations != nullptr && brain.route_incarnations[derivation.route_index] == derivation.route_incarnations[0]) {
      const std::int64_t revised_parameter = static_cast<std::int64_t>(derivation.parameters_q16[0]) + revision.resource_delta;
      if (revised_parameter < -0x80000000ll || revised_parameter > 0x7fffffffll) return false;
      brain.routes[derivation.route_index].conductance_q16 = derivation.parameters_q16[0] = static_cast<std::int32_t>(revised_parameter);
    }
    for (std::uint32_t i = 0u; i < brain.postbirth_constructor->recipe_incidence_count; ++i) {
      auto& incidence = brain.postbirth_constructor->recipe_incidence[i];
      if (incidence.derivation_index == d && incidence.revision_identity == prior_revision)
        incidence.revision_identity = derivation.revision_identity;
    }
    for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
      auto& entry = frontier->entries[slot];
      if (entry.state == ResidentActualFrontierState::live && entry.derivation_index == d &&
          entry.revision_identity == prior_revision && entry.occurrence.revision_identity == prior_revision) {
        entry.revision_identity = derivation.revision_identity;
        entry.occurrence.revision_identity = derivation.revision_identity;
        if (!lower_resident_executable_morphology(brain, derivation, entry.occurrence, false, &entry.work) ||
            !execute_resident_executable_morphology(brain, derivation, entry.occurrence, entry.work, entry.occurrence.activation_q16, &entry.output_q16, &entry.work_units)) return false;
      }
    }
  }
  return true;
}
}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_INVERSE_TRANSFORMATION_CUH
