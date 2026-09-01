#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RECIPE_CREDIT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RECIPE_CREDIT_CUH

#include "direct_adult_core.cuh"
#include "direct_adult_recipe_ir.cuh"
#include "direct_adult_runtime_frontiers.cuh"

namespace substrate::direct_adult_core {

#ifndef DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER
#define DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER __device__ inline
#define DIRECT_ADULT_RECIPE_CREDIT_LOCAL_EXPORT_QUALIFIER 1
#endif

__device__ inline bool resident_world_return_work_current(
    DirectBrain brain, const ResidentActualFrontierEntry& entry);

__device__ inline const ResidentActualFrontierEntry*
resident_recipe_ir_actual_work(
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    const DirectActionParticipationLink& link) {
  if (actual_frontier == nullptr || brain.postbirth_constructor == nullptr ||
      actual_frontier->live_count > kResidentActualFrontierCapacity ||
      link.logical_recipe_id == 0u || link.revision_identity == 0u ||
      link.occurrence_identity == 0u || link.participation_identity == 0u)
    return nullptr;
  const ResidentActualFrontierEntry* match = nullptr;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const auto& entry = actual_frontier->entries[i];
    const auto& occurrence = entry.occurrence;
    if (entry.state != ResidentActualFrontierState::live ||
        occurrence.state != kResidentRecipeOccurrenceLive ||
        occurrence.occurrence_identity != link.occurrence_identity ||
        occurrence.logical_recipe_id != link.logical_recipe_id ||
        occurrence.revision_identity != link.revision_identity ||
        occurrence.participation_identity != link.participation_identity ||
        occurrence.context_signature != link.occurrence_context_signature ||
        occurrence.route_incarnation != link.occurrence_route_incarnation ||
        occurrence.source_incarnation != link.claim_incarnation ||
        occurrence.authority != link.authority)
      continue;
    if (match != nullptr || !resident_world_return_work_current(brain, entry))
      return nullptr;
    match = &entry;
  }
  return match;
}

DIRECT_ADULT_HD inline std::uint64_t resident_recipe_consequence_identity(
    const direct_network::DirectExactHistoryRecord& consequence) {
  using namespace direct_network;
  if (consequence.kind != DirectExactHistoryKind::world_return ||
      consequence.identity == 0u || consequence.parent_identity == 0u ||
      (consequence.flags & kDirectHistoryVerifiedObservation) == 0u ||
      consequence.resource_delta == 0)
    return 0u;
  const std::uint64_t identity = exact_history_fold_record(
      0x7265636977636f6eull, consequence);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_recipe_ir_revision_identity(
    const direct_network::ResidentRecipeCell& cell,
    const direct_network::DirectExactHistoryRecord& event,
    std::int64_t exact_credit_after, std::int64_t policy_parameter_after,
    std::uint64_t binding_identity) {
  std::uint64_t identity = direct_network::resident_recipe_revision_identity(
      cell.logical_recipe_id, cell.revision_identity, cell.revision + 1u,
      direct_network::resident_recipe_revision_contributor(event),
      cell.support_q16, exact_credit_after);
  identity = direct_network::exact_history_fold_word(
      identity, static_cast<std::uint64_t>(policy_parameter_after));
  identity = direct_network::exact_history_fold_word(identity,
                                                      binding_identity);
  identity = direct_network::exact_history_fold_word(
      identity, event.incarnation_after);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool stage_resident_recipe_ir_revision(
    direct_network::DirectExactHistoryRecord* event,
    const direct_network::ResidentRecipeCell& cell, std::uint32_t recipe_cell,
    std::uint32_t tick, std::uint32_t event_tick, std::uint32_t route_index,
    std::int32_t exact_credit_delta_q16,
    const ResidentRecipeIrEvidence& evidence,
    const ResidentRecipeIrResult& result) {
  if (event == nullptr || exact_credit_delta_q16 == 0 ||
      evidence.recipe_cell != recipe_cell ||
      evidence.logical_recipe_id != cell.logical_recipe_id ||
      evidence.prior_revision_identity != cell.revision_identity ||
      evidence.exact_credit_delta_q16 != exact_credit_delta_q16 ||
      evidence.binding_identity == 0u || result.execution_identity == 0u ||
      !resident_recipe_ir_intact(cell.update_program) ||
      evidence.binding_identity != resident_recipe_ir_binding_identity(
          evidence.subject_identity, cell.logical_recipe_id,
          cell.ir_binding_identity == 0u ? cell.revision_identity
                                         : cell.ir_bound_revision_identity,
          cell.update_program.program_identity) ||
      result.execution_identity != resident_recipe_ir_execution_identity(
          cell.update_program, evidence, result.parameter_delta_q16,
          result.work_units) ||
      (exact_credit_delta_q16 > 0 &&
       cell.credit_q16 > INT64_MAX - exact_credit_delta_q16) ||
      (exact_credit_delta_q16 < 0 &&
       cell.credit_q16 < INT64_MIN - exact_credit_delta_q16) ||
      (result.parameter_delta_q16 > 0 &&
       cell.policy_parameter_q16 > INT64_MAX - result.parameter_delta_q16) ||
      (result.parameter_delta_q16 < 0 &&
       cell.policy_parameter_q16 < INT64_MIN - result.parameter_delta_q16))
    return false;
  *event = direct_network::DirectExactHistoryRecord{};
  event->parent_identity = cell.revision_identity;
  event->resident_tick = tick;
  event->event_tick = event_tick;
  event->kind = direct_network::DirectExactHistoryKind::recipe_revision;
  event->source = recipe_cell;
  event->subject = static_cast<std::uint32_t>(
      direct_network::ResidentRecipeRevisionAuthority::resident_ir);
  event->value = result.work_units;
  event->context = route_index;
  event->flags = cell.update_program.abi_version;
  event->incarnation_before = evidence.consequence_identity;
  event->incarnation_after = result.execution_identity;
  event->resource_delta = result.parameter_delta_q16;
  event->identity = resident_recipe_ir_revision_identity(
      cell, *event, cell.credit_q16 + exact_credit_delta_q16,
      cell.policy_parameter_q16 + result.parameter_delta_q16,
      evidence.binding_identity);
  return true;
}

DIRECT_ADULT_HD inline bool apply_resident_recipe_ir_revision(
    direct_network::ResidentRecipeCell* cell,
    const direct_network::DirectExactHistoryRecord& exact_credit,
    const direct_network::DirectExactHistoryRecord& revision,
    std::uint32_t recipe_cell, std::uint64_t subject_identity,
    std::uint64_t binding_identity) {
  using namespace direct_network;
  if (cell == nullptr || exact_credit.kind != DirectExactHistoryKind::recipe_commit ||
      revision.kind != DirectExactHistoryKind::recipe_revision ||
      exact_credit.source != recipe_cell || revision.source != recipe_cell ||
      revision.subject != static_cast<std::uint32_t>(
          ResidentRecipeRevisionAuthority::resident_ir) ||
      revision.parent_identity != cell->revision_identity ||
      exact_credit.incarnation_before != static_cast<std::uint64_t>(cell->credit_q16) ||
      exact_credit.incarnation_after != static_cast<std::uint64_t>(
          cell->credit_q16 + exact_credit.resource_delta) ||
      revision.incarnation_before == 0u || revision.incarnation_after == 0u ||
      subject_identity == 0u || binding_identity == 0u ||
      binding_identity != resident_recipe_ir_binding_identity(
          subject_identity, cell->logical_recipe_id,
          cell->ir_binding_identity == 0u ? cell->revision_identity
                                         : cell->ir_bound_revision_identity,
          cell->update_program.program_identity) ||
      cell->ir_last_execution_identity == revision.incarnation_after ||
      (cell->ir_binding_identity != 0u &&
       (cell->ir_subject_identity != subject_identity ||
        cell->ir_binding_identity != binding_identity)) ||
      revision.identity != resident_recipe_ir_revision_identity(
          *cell, revision, cell->credit_q16 + exact_credit.resource_delta,
          cell->policy_parameter_q16 + revision.resource_delta,
          binding_identity))
    return false;
  if (cell->ir_binding_identity == 0u) {
    cell->ir_subject_identity = subject_identity;
    cell->ir_bound_revision_identity = cell->revision_identity;
    cell->ir_binding_identity = binding_identity;
  }
  cell->credit_q16 += exact_credit.resource_delta;
  cell->policy_parameter_q16 += revision.resource_delta;
  cell->ir_last_consequence_identity = revision.incarnation_before;
  cell->ir_last_execution_identity = revision.incarnation_after;
  ++cell->ir_execution_count;
  cell->revision_identity = revision.identity;
  ++cell->revision;
  set_resident_recipe_current_revision_authority(
      cell, ResidentRecipeRevisionAuthority::resident_ir);
  if (cell->receptor_state.causal_identity != 0u)
    cell->receptor_state.revision_identity = cell->revision_identity;
  return true;
}

// Plans both the exact Recipe credit record and its causally paired immutable
// RecipeRevision record. No Recipe cell changes until the enclosing world-
// return history phase has accepted the complete transaction.
__device__ inline std::uint32_t plan_recipe_credit_transaction(
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    const direct_network::DirectExactHistoryRecord& consequence,
    direct_network::ResidentRecipeCell* recipe_cells, std::uint32_t recipe_cell_count,
    DirectRoute* routes, const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, const std::uint32_t* sparse_links,
    const std::int32_t* exact_credit_deltas_q16,
    std::uint32_t sparse_count, std::uint64_t action_ticket,
    std::uint32_t commit_tick, std::uint32_t emission_tick,
    direct_network::DirectExactHistoryRecord* records, bool* transaction_valid) {
  if (transaction_valid == nullptr) return 0u;
  *transaction_valid = true;
  const std::uint64_t consequence_identity =
      resident_recipe_consequence_identity(consequence);
  std::uint32_t count = 0u, valid_sparse[kMaxActionParticipationLinks]{};
  for (std::uint32_t i = 0u; recipe_cells != nullptr && i < sparse_count; ++i) {
    const auto& link = action_links[participant_offset + sparse_links[i]];
    const std::uint32_t cell = direct_network::decode_route_recipe_builder(routes[link.route_index].flags);
    const std::int32_t delta = exact_credit_deltas_q16[i];
    if (cell >= recipe_cell_count || delta == 0) continue;
    std::int64_t prior = recipe_cells[cell].credit_q16;
    for (std::uint32_t j = 0u; j < count; ++j)
      if (records[j].source == cell)
        prior = static_cast<std::int64_t>(records[j].incarnation_after);
    if ((delta > 0 && prior > INT64_MAX - delta) ||
        (delta < 0 && prior < INT64_MIN - delta)) {
      *transaction_valid = false;
      return 0u;
    }
    valid_sparse[count] = i;
    auto& record = records[count++];
    record.identity = action_ticket; record.parent_identity = link.participation_identity != 0u ? link.participation_identity : link.participant_ticket_id;
    record.resident_tick = commit_tick; record.event_tick = emission_tick;
    record.kind = direct_network::DirectExactHistoryKind::recipe_commit;
    record.source = cell; record.subject = link.route_index; record.value = kInvalidIndex;
    record.context = static_cast<std::uint32_t>(link.route_incarnation);
    record.flags = static_cast<std::uint32_t>(link.route_incarnation >> 32u);
    record.incarnation_before = static_cast<std::uint64_t>(prior);
    record.incarnation_after = static_cast<std::uint64_t>(prior + delta); record.resource_delta = delta;
  }
  const std::uint32_t detail_count = count;
  for (std::uint32_t i = 0u; i < detail_count; ++i) {
    const auto& detail = records[i];
    const auto& link = action_links[participant_offset + sparse_links[valid_sparse[i]]];
    direct_network::ResidentRecipeCell predicted = recipe_cells[detail.source];
    for (std::uint32_t j = 0u; j < i; ++j) {
      if (records[detail_count + j].source != detail.source) continue;
      const auto& prior_revision = records[detail_count + j];
      const auto prior_authority = static_cast<
          direct_network::ResidentRecipeRevisionAuthority>(
          prior_revision.subject);
      bool applied = false;
      if (prior_authority ==
          direct_network::ResidentRecipeRevisionAuthority::resident_ir) {
        const std::uint64_t subject_identity =
            resident_recipe_ir_subject_identity(brain.birth_root);
        std::uint64_t binding_identity = predicted.ir_binding_identity;
        if (binding_identity == 0u)
          binding_identity = resident_recipe_ir_binding_identity(
              subject_identity, predicted);
        applied = apply_resident_recipe_ir_revision(
            &predicted, records[j], prior_revision, detail.source,
            subject_identity, binding_identity);
      } else {
        applied = direct_network::apply_resident_recipe_revision_event(
            &predicted, prior_revision, detail.source);
      }
      if (!applied) {
        *transaction_valid = false;
        return 0u;
      }
    }
    const ResidentActualFrontierEntry* active =
        resident_recipe_ir_actual_work(brain, actual_frontier, link);
    if (active == nullptr) {
      direct_network::stage_resident_recipe_revision_event(
          &records[detail_count + i], predicted, detail.source,
          direct_network::ResidentRecipeRevisionAuthority::experience,
          link.occurrence_identity != 0u ? link.occurrence_identity : action_ticket,
          link.participation_identity != 0u ? link.participation_identity
                                            : link.participant_ticket_id,
          commit_tick, emission_tick, link.route_index, valid_sparse[i],
          static_cast<std::uint32_t>(link.route_incarnation ^
                                     (link.route_incarnation >> 32u)),
          detail.resource_delta);
      continue;
    }
    const std::uint64_t subject_identity =
        resident_recipe_ir_subject_identity(brain.birth_root);
    std::uint64_t binding_identity = predicted.ir_binding_identity;
    if (binding_identity == 0u)
      binding_identity = resident_recipe_ir_binding_identity(subject_identity,
                                                              predicted);
    else if (predicted.ir_subject_identity != subject_identity ||
             binding_identity != resident_recipe_ir_binding_identity(
                 subject_identity, predicted.logical_recipe_id,
                 predicted.ir_bound_revision_identity,
                 predicted.update_program.program_identity)) {
      *transaction_valid = false;
      return 0u;
    }
    ResidentRecipeIrEvidence evidence{};
    evidence.subject_identity = subject_identity;
    evidence.binding_identity = binding_identity;
    evidence.consequence_identity = consequence_identity;
    evidence.occurrence_identity = link.occurrence_identity;
    evidence.participation_identity = link.participation_identity;
    evidence.active_work_identity = active->work.identity;
    evidence.logical_recipe_id = predicted.logical_recipe_id;
    evidence.prior_revision_identity = predicted.revision_identity;
    evidence.exact_credit_delta_q16 = static_cast<std::int32_t>(
        detail.resource_delta);
    evidence.recipe_cell = detail.source;
    ResidentRecipeIrResult result{};
    if (consequence_identity == 0u) {
      *transaction_valid = false;
      return 0u;
    }
    if (!resident_recipe_ir_intact(predicted.update_program)) {
      *transaction_valid = false;
      return 0u;
    }
    if (!execute_resident_recipe_ir(predicted.update_program, evidence,
                                    &result)) {
      *transaction_valid = false;
      return 0u;
    }
    if (!stage_resident_recipe_ir_revision(
            &records[detail_count + i], predicted, detail.source, commit_tick,
            emission_tick, link.route_index,
            static_cast<std::int32_t>(detail.resource_delta), evidence,
            result)) {
      *transaction_valid = false;
      return 0u;
    }
  }
  return 2u * detail_count;
}

struct ResidentWorldReturnRecipeRebindPlan {
  direct_network::ResidentRecipeCell cells[kMaxActionParticipationLinks]{};
  std::uint32_t cell_indices[kMaxActionParticipationLinks]{};
  std::uint64_t derivation_initial_revision[
      direct_network::kResidentPostbirthRecipeReserve]{};
  std::uint64_t derivation_final_revision[
      direct_network::kResidentPostbirthRecipeReserve]{};
  std::int32_t derivation_final_parameter_q16[
      direct_network::kResidentPostbirthRecipeReserve]{};
  ResidentActualFrontierEntry frontier_entries[
      kResidentActualFrontierCapacity]{};
  bool derivation_slots[direct_network::kResidentPostbirthRecipeReserve]{};
  bool derivation_parameter_slots[
      direct_network::kResidentPostbirthRecipeReserve]{};
  bool frontier_slots[kResidentActualFrontierCapacity]{};
  std::uint32_t cell_count{};
};

__device__ inline bool resident_world_return_work_current(
    DirectBrain brain, const ResidentActualFrontierEntry& entry) {
  using namespace direct_network;
  if (entry.derivation_index >=
          brain.postbirth_constructor->derivation_count ||
      entry.work.logical_recipe_id != entry.occurrence.logical_recipe_id ||
      entry.work.occurrence_identity !=
          entry.occurrence.occurrence_identity ||
      entry.work.participation_identity !=
          entry.occurrence.participation_identity ||
      entry.work.context_signature != entry.occurrence.context_signature ||
      entry.work.binding_count != entry.occurrence.binding_count ||
      entry.work.binding_count > kResidentDerivationWidth ||
      entry.work.route_count > 2u || entry.work.identity == 0u ||
      entry.work.identity != resident_executable_work_identity(entry.work) ||
      !resident_actual_frontier_entry_current(
          brain, *brain.postbirth_constructor, entry))
    return false;
  for (std::uint32_t binding = 0u;
       binding < entry.work.binding_count; ++binding)
    if (entry.work.variable_identities[binding] !=
        entry.occurrence.bindings[binding].variable_identity)
      return false;
  for (std::uint32_t route = 0u; route < entry.work.route_count; ++route) {
    const std::uint32_t route_index = entry.work.route_indices[route];
    if (brain.route_incarnations == nullptr ||
        route_index >= brain.route_capacity ||
        !route_is_active(brain.routes[route_index]) ||
        brain.route_incarnations[route_index] !=
            entry.work.route_incarnations[route] ||
        brain.routes[route_index].conductance_q16 !=
            entry.work.conductances_q16[route])
      return false;
  }
  return true;
}

// Stage the exact post-credit Recipe/live-Occurrence state without touching
// authoritative storage. Sparse route updates have not published yet, so each
// cached dependency is advanced by its exact recorded delta here. Condensed
// representations refuse because their frozen proof revisions cannot be
// rewritten as if a new witness existed.
__device__ inline bool stage_world_return_recipe_rebind(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    const direct_network::DirectExactHistoryRecord& consequence,
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count,
    ResidentWorldReturnRecipeRebindPlan* plan) {
  using namespace direct_network;
  if (plan == nullptr || records == nullptr) return false; *plan = ResidentWorldReturnRecipeRebindPlan{};
  // Empty Recipe credit is a valid no-op rebind.
  if (record_count == 0u) return true;
  const bool has_constructor = brain.postbirth_constructor != nullptr, has_derivations = brain.postbirth_derivations != nullptr;
  if (has_constructor != has_derivations ||
      brain.development == nullptr || brain.recipe_cells == nullptr ||
      (has_constructor && actual_frontier == nullptr) ||
      record_count % 2u != 0u ||
      record_count / 2u > kMaxActionParticipationLinks ||
      (has_constructor && (brain.postbirth_constructor->derivation_count > kResidentPostbirthRecipeReserve ||
                           brain.postbirth_constructor->recipe_incidence_count > kResidentRecipeIncidenceCapacity ||
                           actual_frontier->live_count > kResidentActualFrontierCapacity)))
    return false;
  const std::uint32_t detail_count = record_count / 2u;
  const std::uint64_t consequence_identity =
      resident_recipe_consequence_identity(consequence);
  for (std::uint32_t i = 0u; i < detail_count; ++i) {
    const auto& detail = records[i];
    const auto& revision = records[detail_count + i];
    const auto authority = static_cast<ResidentRecipeRevisionAuthority>(
        revision.subject);
    if (detail.kind != DirectExactHistoryKind::recipe_commit ||
        revision.kind != DirectExactHistoryKind::recipe_revision ||
        detail.source != revision.source ||
        (authority == ResidentRecipeRevisionAuthority::experience &&
         detail.resource_delta != revision.resource_delta) ||
        (authority != ResidentRecipeRevisionAuthority::experience &&
         authority != ResidentRecipeRevisionAuthority::resident_ir) ||
        revision.source >= brain.development->recipe_cell_count)
      return false;
    std::uint32_t cell_slot = plan->cell_count;
    for (std::uint32_t c = 0u; c < plan->cell_count; ++c)
      if (plan->cell_indices[c] == revision.source) cell_slot = c;
    if (cell_slot == plan->cell_count) {
      if (plan->cell_count >= kMaxActionParticipationLinks) return false;
      plan->cell_indices[cell_slot] = revision.source;
      plan->cells[cell_slot] = brain.recipe_cells[revision.source];
      ++plan->cell_count;
    }
    if (plan->cells[cell_slot].revision_identity != revision.parent_identity)
      return false;
    if (authority == ResidentRecipeRevisionAuthority::experience) {
      if (!apply_resident_recipe_revision_event(
              &plan->cells[cell_slot], revision, revision.source))
        return false;
    } else {
      auto& cell = plan->cells[cell_slot];
      const std::uint64_t subject_identity =
          resident_recipe_ir_subject_identity(brain.birth_root);
      std::uint64_t binding_identity = cell.ir_binding_identity;
      if (binding_identity == 0u)
        binding_identity = resident_recipe_ir_binding_identity(subject_identity,
                                                                cell);
      else if (cell.ir_subject_identity != subject_identity ||
               binding_identity != resident_recipe_ir_binding_identity(
                   subject_identity, cell.logical_recipe_id,
                   cell.ir_bound_revision_identity,
                   cell.update_program.program_identity))
        return false;
      if (consequence_identity == 0u ||
          revision.incarnation_before != consequence_identity ||
          !apply_resident_recipe_ir_revision(
              &cell, detail, revision, revision.source, subject_identity,
              binding_identity))
        return false;
    }

    if (!has_constructor) continue;
    std::uint32_t derivation_matches = 0u;
    for (std::uint32_t d = 0u; d < brain.postbirth_constructor->derivation_count; ++d) {
      const auto& derivation = brain.postbirth_derivations[d];
      if (derivation.recipe_cell != revision.source) continue;
      ++derivation_matches;
      const std::uint64_t prior_revision = plan->derivation_slots[d]
          ? plan->derivation_final_revision[d]
          : derivation.revision_identity;
      if (derivation.logical_recipe_id !=
              plan->cells[cell_slot].logical_recipe_id ||
          prior_revision != revision.parent_identity ||
          (derivation.condensation_flags &
           kResidentDerivationCondensedNetwork) != 0u)
        return false;
      if (!plan->derivation_slots[d]) {
        plan->derivation_slots[d] = true;
        plan->derivation_initial_revision[d] = derivation.revision_identity;
        plan->derivation_final_parameter_q16[d] =
            derivation.parameters_q16[0];
      }
      const bool route_gain = derivation.relation_count == 1u &&
          derivation.parameter_count == 1u &&
          derivation.relations[0] == static_cast<std::uint32_t>(
              ResidentRecipeRelation::route_gain) &&
          derivation.route_index == detail.subject;
      if (route_gain) {
        const std::uint64_t route_incarnation =
            static_cast<std::uint64_t>(detail.context) |
            (static_cast<std::uint64_t>(detail.flags) << 32u);
        const std::int64_t parameter = static_cast<std::int64_t>(
            plan->derivation_final_parameter_q16[d]) +
            revision.resource_delta;
        if (brain.route_incarnations == nullptr ||
            detail.subject >= brain.route_capacity ||
            derivation.route_incarnations[0] != route_incarnation ||
            brain.route_incarnations[detail.subject] != route_incarnation ||
            !route_is_active(brain.routes[detail.subject]) ||
            parameter < INT32_MIN || parameter > INT32_MAX)
          return false;
        plan->derivation_parameter_slots[d] = true;
        plan->derivation_final_parameter_q16[d] =
            static_cast<std::int32_t>(parameter);
      }
      plan->derivation_final_revision[d] =
          plan->cells[cell_slot].revision_identity;

      for (std::uint32_t slot = 0u;
           slot < kResidentActualFrontierCapacity; ++slot) {
        const auto& current = actual_frontier->entries[slot];
        if (current.state != ResidentActualFrontierState::live ||
            current.derivation_index != d)
          continue;
        auto& rebound = plan->frontier_entries[slot];
        if (!plan->frontier_slots[slot]) {
          if (current.revision_identity != derivation.revision_identity ||
              current.occurrence.revision_identity !=
                  derivation.revision_identity ||
              current.work.revision_identity != derivation.revision_identity ||
              !resident_world_return_work_current(brain, current))
            return false;
          plan->frontier_slots[slot] = true;
          rebound = current;
        }
        if (rebound.revision_identity != revision.parent_identity ||
            rebound.occurrence.revision_identity !=
                revision.parent_identity ||
            rebound.work.revision_identity != revision.parent_identity)
          return false;
        rebound.revision_identity = plan->cells[cell_slot].revision_identity;
        rebound.occurrence.revision_identity =
            plan->cells[cell_slot].revision_identity;
        rebound.work.revision_identity =
            plan->cells[cell_slot].revision_identity;
        rebound.work.identity = resident_executable_work_identity(rebound.work);
      }
    }
    if (derivation_matches == 0u) return false;
  }
  if (!has_constructor) return true;
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
    const auto& current = actual_frontier->entries[slot];
    if (current.state != ResidentActualFrontierState::live) continue;
    bool route_affected = false;
    for (std::uint32_t route = 0u; route < current.work.route_count; ++route)
      for (std::uint32_t detail = 0u; detail < detail_count; ++detail)
        route_affected |= current.work.route_indices[route] ==
            records[detail].subject;
    if (!route_affected) continue;
    if (!plan->frontier_slots[slot]) {
      if (!resident_world_return_work_current(brain, current)) return false;
      plan->frontier_slots[slot] = true;
      plan->frontier_entries[slot] = current;
    }
    auto& rebound = plan->frontier_entries[slot];
    for (std::uint32_t detail = 0u; detail < detail_count; ++detail)
      for (std::uint32_t route = 0u;
           route < rebound.work.route_count; ++route)
        if (rebound.work.route_indices[route] == records[detail].subject) {
          const std::int64_t conductance = static_cast<std::int64_t>(
              rebound.work.conductances_q16[route]) +
              records[detail].resource_delta;
          if (conductance < INT32_MIN || conductance > INT32_MAX)
            return false;
          rebound.work.conductances_q16[route] =
              static_cast<std::int32_t>(conductance);
        }
  }
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
    if (!plan->frontier_slots[slot]) continue;
    auto& rebound = plan->frontier_entries[slot];
    const std::uint32_t d = rebound.derivation_index;
    const auto& derivation = brain.postbirth_derivations[d];
    if (derivation.relation_count == 1u &&
        derivation.parameter_count == 1u) {
      if (rebound.work.route_count != 1u ||
          rebound.work.route_indices[0] != derivation.route_index ||
          !evaluate_resident_recipe_boundary_q16(
              derivation.relations[0],
              plan->derivation_slots[d]
                  ? plan->derivation_final_parameter_q16[d]
                  : derivation.parameters_q16[0],
              rebound.occurrence.activation_q16, &rebound.output_q16))
        return false;
      rebound.work.work_class =
          ResidentExecutableWorkClass::generic_route_micrograph;
      rebound.work_units = 1u;
    } else {
      if (derivation.relation_count != 2u ||
          derivation.parameter_count != 4u ||
          derivation.parameters_q16[3] != 2 ||
          rebound.work.route_count != 2u ||
          rebound.work.route_indices[0] != derivation.relations[0] ||
          rebound.work.route_indices[1] != derivation.relations[1])
        return false;
      const bool compiled =
          rebound.work.conductances_q16[0] == derivation.parameters_q16[1] &&
          rebound.work.conductances_q16[1] == derivation.parameters_q16[2];
      rebound.work.work_class = compiled
          ? ResidentExecutableWorkClass::compiled_macro
          : ResidentExecutableWorkClass::generic_route_micrograph;
      std::int64_t value = rebound.occurrence.activation_q16;
      const std::int32_t composed = static_cast<std::int32_t>(
          (static_cast<std::int64_t>(derivation.parameters_q16[1]) *
           derivation.parameters_q16[2]) >> 16);
      if (compiled && derivation.parameters_q16[0] == composed) {
        value = (value * derivation.parameters_q16[0]) >> 16;
        rebound.work_units = 1u;
      } else {
        value = (value * rebound.work.conductances_q16[0]) >> 16;
        value = (value * rebound.work.conductances_q16[1]) >> 16;
        rebound.work_units = 2u;
      }
      if (value < INT32_MIN || value > INT32_MAX) return false;
      rebound.output_q16 = static_cast<std::int32_t>(value);
    }
    rebound.work.identity = resident_executable_work_identity(rebound.work);
  }
  for (std::uint32_t i = 0u;
       i < brain.postbirth_constructor->recipe_incidence_count; ++i) {
    const auto& incidence = brain.postbirth_constructor->recipe_incidence[i];
    if (incidence.derivation_index < kResidentPostbirthRecipeReserve &&
        plan->derivation_slots[incidence.derivation_index] &&
        incidence.revision_identity !=
            plan->derivation_initial_revision[incidence.derivation_index])
      return false;
  }
  return true;
}

__device__ inline void publish_world_return_recipe_rebind(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    const ResidentWorldReturnRecipeRebindPlan& plan) {
  using namespace direct_network;
  if (plan.cell_count == 0u) return;
  for (std::uint32_t c = 0u; c < plan.cell_count; ++c)
    brain.recipe_cells[plan.cell_indices[c]] = plan.cells[c];
  if (brain.postbirth_constructor == nullptr || brain.postbirth_derivations == nullptr) return;
  for (std::uint32_t d = 0u; d < brain.postbirth_constructor->derivation_count; ++d) {
    if (!plan.derivation_slots[d]) continue;
    auto& derivation = brain.postbirth_derivations[d];
    derivation.revision_identity = plan.derivation_final_revision[d];
    if (plan.derivation_parameter_slots[d])
      derivation.parameters_q16[0] = plan.derivation_final_parameter_q16[d];
  }
  for (std::uint32_t i = 0u;
       i < brain.postbirth_constructor->recipe_incidence_count; ++i) {
    auto& incidence = brain.postbirth_constructor->recipe_incidence[i];
    if (incidence.derivation_index < kResidentPostbirthRecipeReserve &&
        plan.derivation_slots[incidence.derivation_index] &&
        incidence.revision_identity ==
            plan.derivation_initial_revision[incidence.derivation_index])
      incidence.revision_identity =
          plan.derivation_final_revision[incidence.derivation_index];
  }
  for (std::uint32_t slot = 0u;
       slot < kResidentActualFrontierCapacity; ++slot)
    if (plan.frontier_slots[slot])
      actual_frontier->entries[slot] = plan.frontier_entries[slot];
}

// Commit already-formed mismatch assignments one bounded receipt at a time.
// The mismatch observation and paired Recipe/RecipeRevision records share one
// exact-history phase; no Recipe state changes unless all three records fit.
DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER std::uint32_t
commit_resident_mismatch_credit_receipts(
    DirectBrain brain, ResidentMismatchOmissionFrontier* frontier,
    std::uint32_t current_tick,
    ResidentActualFrontier* actual_frontier) {
  using namespace direct_network;
  if (frontier == nullptr || brain.development == nullptr ||
      brain.recipe_cells == nullptr || brain.routes == nullptr ||
      brain.route_incarnations == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      actual_frontier == nullptr ||
      frontier->receipt_count > kResidentMismatchReceiptCapacity ||
      frontier->committed_receipt_count > frontier->receipt_count)
    return 0u;
  auto* history = &brain.development->exact_history;
  std::uint32_t committed = 0u;
  while (frontier->committed_receipt_count < frontier->receipt_count) {
    auto& receipt =
        frontier->receipts[frontier->committed_receipt_count];
    if (current_tick <= receipt.horizon_tick) break;
    if (receipt.target_route_index >= brain.route_capacity ||
        receipt.target_route_incarnation == 0u ||
        brain.route_incarnations[receipt.target_route_index] !=
            receipt.target_route_incarnation ||
        !route_is_active(brain.routes[receipt.target_route_index]) ||
        receipt.target_occurrence_identity == 0u ||
        receipt.target_participation_identity == 0u ||
        receipt.target_logical_recipe_id == 0u ||
        receipt.target_revision_identity == 0u ||
        receipt.target_source_identity == 0u ||
        receipt.target_source_incarnation == 0u ||
        receipt.causal_credit_delta_q16 == 0 ||
        receipt.target_derivation_index >=
            brain.postbirth_constructor->derivation_count) {
      ++frontier->refusals;
      break;
    }
    auto& derivation =
        brain.postbirth_derivations[receipt.target_derivation_index];
    const std::uint32_t cell_index = receipt.target_recipe_cell;
    if (cell_index >= brain.development->recipe_cell_count ||
        derivation.recipe_cell != cell_index ||
        derivation.route_index != receipt.target_route_index ||
        derivation.logical_recipe_id != receipt.target_logical_recipe_id) {
      ++frontier->refusals;
      break;
    }
    auto& cell = brain.recipe_cells[cell_index];
    bool revision_current =
        cell.revision_identity == receipt.target_revision_identity;
    for (std::uint32_t prior = frontier->committed_receipt_count;
         !revision_current && prior != 0u; --prior) {
      const auto& ancestor = frontier->receipts[prior - 1u];
      revision_current = ancestor.target_recipe_cell == cell_index &&
                         ancestor.target_logical_recipe_id ==
                             receipt.target_logical_recipe_id &&
                         ancestor.target_revision_identity ==
                             receipt.target_revision_identity &&
                         ancestor.committed_revision_identity ==
                             cell.revision_identity;
    }
    if (cell.logical_recipe_id != receipt.target_logical_recipe_id ||
        derivation.revision_identity != cell.revision_identity ||
        !revision_current ||
        (receipt.causal_credit_delta_q16 > 0 &&
         cell.credit_q16 > INT64_MAX - receipt.causal_credit_delta_q16) ||
        (receipt.causal_credit_delta_q16 < 0 &&
         cell.credit_q16 < INT64_MIN - receipt.causal_credit_delta_q16)) {
      ++frontier->refusals;
      break;
    }
    DirectExactHistoryRecord records[3]{};
    const DirectExactHistoryKind mismatch_kind =
        receipt.kind == ResidentMismatchKind::unexpected_presence
            ? DirectExactHistoryKind::unexpected_presence
            : DirectExactHistoryKind::omitted_consequence;
    stage_mismatch_credit_history_record(
        records, mismatch_kind, receipt.identity,
        receipt.expectation_identity != 0u ? receipt.expectation_identity
                                            : receipt.actual_occurrence_identity,
        receipt.resident_tick, receipt.horizon_tick, receipt.target_route_index,
        receipt.target_context_signature, receipt.target_source_incarnation,
        receipt.target_route_incarnation, receipt.causal_credit_delta_q16,
        receipt.kind == ResidentMismatchKind::unexpected_presence);
    records[1].identity = receipt.identity;
    records[1].parent_identity = receipt.target_participation_identity;
    records[1].resident_tick = receipt.resident_tick;
    records[1].event_tick = receipt.horizon_tick;
    records[1].kind = DirectExactHistoryKind::recipe_commit;
    records[1].source = cell_index;
    records[1].subject = receipt.target_route_index;
    records[1].context = static_cast<std::uint32_t>(
        receipt.target_route_incarnation);
    records[1].flags = static_cast<std::uint32_t>(
        receipt.target_route_incarnation >> 32u);
    records[1].incarnation_before = static_cast<std::uint64_t>(cell.credit_q16);
    records[1].incarnation_after = static_cast<std::uint64_t>(
        cell.credit_q16 + receipt.causal_credit_delta_q16);
    records[1].resource_delta = receipt.causal_credit_delta_q16;
    stage_resident_recipe_revision_event(
        records + 2u, cell, cell_index,
        ResidentRecipeRevisionAuthority::experience,
        receipt.target_occurrence_identity,
        receipt.target_participation_identity, receipt.resident_tick,
        receipt.horizon_tick, receipt.target_route_index,
        receipt.target_derivation_index, receipt.target_context_signature,
        receipt.causal_credit_delta_q16);

    ResidentRecipeCell predicted_cell = cell;
    bool derivation_slot[kResidentPostbirthRecipeReserve]{};
    ResidentActualFrontierEntry rebound[kResidentActualFrontierCapacity]{};
    bool rebound_slot[kResidentActualFrontierCapacity]{};
    std::uint32_t derivation_count = 0u, rebound_count = 0u;
    std::uint32_t target_matches = 0u;
    bool rebound_valid = apply_resident_recipe_revision_event(
        &predicted_cell, records[2], cell_index);
    for (std::uint32_t d = 0u;
         rebound_valid && d < brain.postbirth_constructor->derivation_count;
         ++d) {
      const auto& candidate = brain.postbirth_derivations[d];
      if (candidate.recipe_cell != cell_index) continue;
      rebound_valid = candidate.logical_recipe_id == cell.logical_recipe_id &&
          candidate.revision_identity == cell.revision_identity &&
          (candidate.condensation_flags &
           kResidentDerivationCondensedNetwork) == 0u;
      derivation_slot[d] = true;
      ++derivation_count;
    }
    rebound_valid &= derivation_count != 0u &&
        derivation_slot[receipt.target_derivation_index];
    if (rebound_valid) {
      rebound_valid = actual_frontier->live_count <=
          kResidentActualFrontierCapacity;
      for (std::uint32_t i = 0u;
           rebound_valid && i < kResidentActualFrontierCapacity; ++i) {
        const auto& entry = actual_frontier->entries[i];
        if (entry.state != ResidentActualFrontierState::live ||
            entry.derivation_index >= kResidentPostbirthRecipeReserve ||
            !derivation_slot[entry.derivation_index])
          continue;
        const bool generic = entry.work.work_class ==
            ResidentExecutableWorkClass::generic_route_micrograph;
        const bool compiled = entry.work.work_class ==
            ResidentExecutableWorkClass::compiled_macro;
        rebound_valid = entry.revision_identity == cell.revision_identity &&
            entry.occurrence.revision_identity == cell.revision_identity &&
            entry.work.revision_identity == cell.revision_identity &&
            entry.work.logical_recipe_id == entry.occurrence.logical_recipe_id &&
            entry.work.occurrence_identity ==
                entry.occurrence.occurrence_identity &&
            entry.work.participation_identity ==
                entry.occurrence.participation_identity &&
            entry.work.context_signature == entry.occurrence.context_signature &&
            entry.work.binding_count == entry.occurrence.binding_count &&
            entry.work.binding_count <= kResidentDerivationWidth &&
            entry.work.route_count <= 2u && (generic || compiled) &&
            (!generic || entry.work.route_count != 0u) &&
            entry.work.identity != 0u &&
            entry.work.identity == resident_executable_work_identity(entry.work) &&
            resident_actual_frontier_entry_current(
                brain, *brain.postbirth_constructor, entry);
        for (std::uint32_t binding = 0u;
             rebound_valid && binding < entry.work.binding_count; ++binding)
          rebound_valid = entry.work.variable_identities[binding] ==
              entry.occurrence.bindings[binding].variable_identity;
        for (std::uint32_t route = 0u;
             rebound_valid && route < entry.work.route_count; ++route) {
          const std::uint32_t route_index = entry.work.route_indices[route];
          rebound_valid = route_index < brain.route_capacity &&
              route_is_active(brain.routes[route_index]) &&
              brain.route_incarnations[route_index] ==
                  entry.work.route_incarnations[route] &&
              brain.routes[route_index].conductance_q16 ==
                  entry.work.conductances_q16[route];
        }
        target_matches +=
            entry.occurrence.occurrence_identity ==
                    receipt.target_occurrence_identity &&
                entry.occurrence.participation_identity ==
                    receipt.target_participation_identity &&
                entry.occurrence.logical_recipe_id ==
                    receipt.target_logical_recipe_id &&
                entry.occurrence.source_identity ==
                    receipt.target_source_identity &&
                entry.occurrence.source_incarnation ==
                    receipt.target_source_incarnation &&
                entry.occurrence.context_signature ==
                    receipt.target_context_signature &&
                entry.occurrence.route_incarnation ==
                    receipt.target_route_incarnation
                ? 1u
                : 0u;
        rebound_slot[i] = true;
        rebound[i] = entry;
        rebound[i].revision_identity = predicted_cell.revision_identity;
        rebound[i].occurrence.revision_identity =
            predicted_cell.revision_identity;
        rebound[i].work.revision_identity = predicted_cell.revision_identity;
        rebound[i].work.identity =
            resident_executable_work_identity(rebound[i].work);
        ++rebound_count;
      }
      rebound_valid &= rebound_count != 0u && target_matches == 1u;
    }
    if (!rebound_valid) {
      ++frontier->refusals;
      break;
    }
    if (history->sealed ||
        history->phase_kind != DirectExactHistoryKind::empty ||
        3u > kDirectExactHistoryHotPageCapacity - history->committed_slots ||
        !begin_exact_history_phase(history, mismatch_kind, 3u,
                                   receipt.resident_tick))
      break;
    for (std::uint32_t i = 0u; i < 3u; ++i)
      history->records[history->phase_base + i] = records[i];
    if (finish_exact_history_phase(history) != 3u) break;
    const std::uint64_t prior_revision_identity = cell.revision_identity;
    cell = predicted_cell;
    for (std::uint32_t d = 0u;
         d < brain.postbirth_constructor->derivation_count; ++d)
      if (derivation_slot[d])
        brain.postbirth_derivations[d].revision_identity =
            predicted_cell.revision_identity;
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
      if (rebound_slot[i]) actual_frontier->entries[i] = rebound[i];
    if (brain.postbirth_constructor->recipe_incidence_count <=
        kResidentRecipeIncidenceCapacity)
      for (std::uint32_t i = 0u;
           i < brain.postbirth_constructor->recipe_incidence_count;
           ++i) {
        auto& incidence =
            brain.postbirth_constructor->recipe_incidence[i];
        if (incidence.derivation_index < kResidentPostbirthRecipeReserve &&
            derivation_slot[incidence.derivation_index] &&
            incidence.revision_identity == prior_revision_identity)
          incidence.revision_identity = cell.revision_identity;
      }
    for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i) {
      auto& expected = frontier->expectations[i];
      if (expected.state == ResidentMismatchExpectationState::pending &&
          expected.recipe_cell == cell_index &&
          expected.derivation_index < kResidentPostbirthRecipeReserve &&
          derivation_slot[expected.derivation_index] &&
          expected.current_revision_identity == prior_revision_identity)
        expected.current_revision_identity = cell.revision_identity;
    }
    receipt.committed_revision_identity = cell.revision_identity;
    ++frontier->committed_receipt_count;
    ++committed;
  }
  return committed;
}

#ifdef DIRECT_ADULT_RECIPE_CREDIT_LOCAL_EXPORT_QUALIFIER
#undef DIRECT_ADULT_RECIPE_CREDIT_LOCAL_EXPORT_QUALIFIER
#undef DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER
#endif

}  // namespace substrate::direct_adult_core

#endif
