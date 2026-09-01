#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTUAL_FRONTIER_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTUAL_FRONTIER_CUH

// Live Occurrence working set for Network observation. Sized to the Network
// Occurrence bound (8) so several coupled coalitions plus isolates can
// share one frontier (#1641). Not tied to per-node provenance slot count.
inline constexpr std::uint32_t kResidentActualFrontierCapacity = 8u;

enum class ResidentActualFrontierState : std::uint32_t {
  free = 0u, live = 1u, settled = 2u,
};

enum class ResidentInverseTransformationState : std::uint32_t {
  free = 0u, proposed = 1u, committed = 2u,
};

struct alignas(8) ResidentInverseTransformationProposal {
  std::uint64_t identity, action_ticket, occurrence_identity;
  std::uint64_t relation_revision_identity, predecessor_identity;
  std::uint64_t proposed_revision_identity;
  std::uint64_t observed_source_identity, observed_source_incarnation;
  std::uint64_t observed_context_signature;
  std::int64_t inferred_theta_q16, residual_q16;
  std::uint32_t recipe_cell, observed_word, proposal_tick, solver_kind;
  std::uint32_t expiry_tick;
  ResidentInverseTransformationState state;
};
static_assert(std::is_standard_layout_v<ResidentInverseTransformationProposal> &&
              std::is_trivial_v<ResidentInverseTransformationProposal> &&
              std::has_unique_object_representations_v<ResidentInverseTransformationProposal>);

struct alignas(8) ResidentActualFrontierEntry {
  ResidentRecipeOccurrence occurrence;
  ResidentExecutableMorphologyWork work;
  std::uint64_t contact_identity, ingress_sequence;
  std::uint64_t revision_identity, route_incarnation;
  std::int32_t output_q16;
  std::uint32_t work_units;
  ResidentActualFrontierState state;
  std::uint32_t derivation_index, nomination_work_units, composition_depth;
  ResidentInverseTransformationProposal inverse;
};
static_assert(std::is_standard_layout_v<ResidentActualFrontierEntry> &&
              std::is_trivial_v<ResidentActualFrontierEntry> &&
              std::has_unique_object_representations_v<ResidentActualFrontierEntry>);

enum class ResidentPendingActualContactState : std::uint32_t {
  free = 0u, pending = 1u, expired = 2u, admitted = 3u,
};
struct alignas(8) ResidentPendingActualContact {
  ActivityEvent event;
  ResidentContactEpochReceipt receipt;
  std::uint32_t claim_incarnation, authority_incarnation, expiry_tick;
  std::uint32_t derivation_ceiling;
  ResidentPendingActualContactState state;
  std::uint32_t reserved;
};
static_assert(sizeof(ResidentPendingActualContact) == 128 &&
              std::is_standard_layout_v<ResidentPendingActualContact> &&
              std::is_trivial_v<ResidentPendingActualContact> &&
              std::has_unique_object_representations_v<ResidentPendingActualContact>);

struct alignas(8) ResidentActualFrontier {
  ResidentActualFrontierEntry entries[kResidentActualFrontierCapacity];
  ResidentPendingActualContact pending_contacts[kResidentActualFrontierCapacity];
  std::uint64_t boundary_session_epoch, last_ingress_sequence;
  std::uint32_t live_count, admissions, refusals, consumed_credentials;
  std::uint32_t pending_contact_count, expired_pending_contacts;
  std::uint32_t condensation_nominations, condensation_promotions;
  std::uint32_t condensed_executions, condensation_deoptimizations;
};
static_assert(std::is_standard_layout_v<ResidentActualFrontier> &&
              std::is_trivial_v<ResidentActualFrontier> &&
              std::has_unique_object_representations_v<ResidentActualFrontier>);
#include "hardware_native/direct_occurrence_activation_soa.cuh"
#include "hardware_native/direct_adult_contact_epoch_identity.cuh"

DIRECT_ADULT_HD inline std::uint32_t expire_resident_actual_frontier(
    ResidentActualFrontier* frontier, std::uint32_t current_tick,
    DirectActualFrontierCausalCounters* counters = nullptr) {
  if (frontier == nullptr) return 0u;
  std::uint32_t live = 0u, pending = 0u;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    auto& entry = frontier->entries[i];
    if (entry.state == ResidentActualFrontierState::live &&
        entry.occurrence.expiry_tick < current_tick) {
      entry.state = ResidentActualFrontierState::settled;
      entry.occurrence.state = kResidentRecipeOccurrenceSettled;
    }
    live += entry.state == ResidentActualFrontierState::live ? 1u : 0u;

    auto& contact = frontier->pending_contacts[i];
    if ((contact.state == ResidentPendingActualContactState::pending ||
         contact.state == ResidentPendingActualContactState::admitted) &&
        contact.expiry_tick < current_tick) {
      contact.state = ResidentPendingActualContactState::expired;
      ++frontier->expired_pending_contacts;
      ++frontier->refusals;
      if (counters != nullptr) ++counters->stale_currentness_rejects;
    }
    pending += contact.state == ResidentPendingActualContactState::pending ? 1u : 0u;
  }
  frontier->live_count = live;
  frontier->pending_contact_count = pending;
  return live;
}

DIRECT_ADULT_HD inline bool resident_contact_credential_valid(
    const DirectBrain& brain, const ActivityEvent& event,
    const ResidentContactEpochReceipt& receipt, std::uint64_t expected_sequence) {
  if (brain.boundary_ports == nullptr || receipt.identity == 0u ||
      event.origin != CausalOrigin::external_contact ||
      receipt.consumed != 0u || receipt.reserved != 0u ||
      receipt.identity != resident_contact_receipt_identity(receipt) ||
      receipt.payload_identity != resident_contact_payload_identity(event) ||
      receipt.ingress_sequence != expected_sequence ||
      receipt.port_index >= brain.boundary_port_count ||
      receipt.selection != ResidentContactSelection::resident_owned ||
      receipt.integration != ResidentContactIntegration::canonical ||
      receipt.source_available != 1u)
    return false;
  const DirectBoundaryPort& port = brain.boundary_ports[receipt.port_index];
  return port.node == event.node && port.channel == event.channel &&
      (port.role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::sensor)) != 0u &&
      receipt.source_identity == resident_contact_source_identity(brain, port) &&
      receipt.codec_identity == resident_contact_codec_identity(brain, port) &&
      receipt.boundary_session_epoch == resident_contact_session_identity(brain, port);
}

DIRECT_ADULT_HD inline bool admit_resident_actual_frontier_contact(
    const DirectBrain& brain, const ActivityEvent& event,
    ResidentContactEpochReceipt* receipt, std::uint64_t expected_sequence,
    const NodeCausalParticipation* active_participation,
    std::uint32_t current_tick, std::uint32_t lifetime_ticks,
    ResidentActualFrontier* frontier,
    DirectActualFrontierCausalCounters* counters = nullptr) {
  using namespace direct_network;
  if (frontier == nullptr || receipt == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr || brain.postbirth_derivations == nullptr ||
      brain.recipe_cells == nullptr || event.ticket_id == 0u ||
      event.timestamp > current_tick || lifetime_ticks == 0u ||
      !resident_contact_credential_valid(brain, event, *receipt, expected_sequence)) {
    if (counters != nullptr && receipt != nullptr &&
        (event.timestamp > current_tick || brain.boundary_ports != nullptr))
      ++counters->stale_currentness_rejects;
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }
  NodeCausalParticipation participation{};
  std::uint32_t participation_matches = 0u;
  if (active_participation != nullptr && event.node < brain.node_count) {
    const auto* base =
        active_participation + event.node * kNodeParticipationAperture;
    for (std::uint32_t i = 0u; i < kNodeParticipationAperture; ++i) {
      const NodeCausalParticipation candidate = base[i];
      if (candidate.ticket_id == event.ticket_id &&
          candidate.claim_incarnation != 0u &&
          candidate.authority == DirectParticipationAuthority::independent_external &&
          candidate.authority_incarnation ==
              resident_contact_authority_incarnation(*receipt, event) &&
          candidate.expiry_tick >= current_tick) {
        participation = candidate;
        ++participation_matches;
      }
    }
  }
  const auto* state = brain.postbirth_constructor;
  if (state->recipe_incidence_count > kResidentRecipeIncidenceCapacity) {
    ++frontier->refusals;
    if (counters != nullptr) ++counters->ambiguity_rejects;
    return false;
  }
  const ResidentRawContactKey contact = resident_raw_contact_key(
      event.node, event.channel, event.word, event.context);
  ResidentSensoryAuthorityReceipt sensory_authority{};
  const DirectExactHistoryHotPage& sensory_history =
      brain.development->exact_history;
  if (sensory_history.phase_kind != DirectExactHistoryKind::empty ||
      sensory_history.phase_admitted != 0u ||
      sensory_history.committed_slots > kDirectExactHistoryHotPageCapacity ||
      sensory_history.last_phase_records == 0u ||
      sensory_history.last_phase_records > sensory_history.committed_slots) {
    ++frontier->refusals;
    return false;
  }
  const DirectExactHistoryRecord* sensory_record = nullptr;
  std::uint32_t sensory_matches = 0u;
  const std::uint32_t sensory_phase_begin =
      sensory_history.committed_slots - sensory_history.last_phase_records;
  for (std::uint32_t i = sensory_phase_begin;
       i < sensory_history.committed_slots; ++i) {
    const DirectExactHistoryRecord& candidate = sensory_history.records[i];
    if (candidate.kind != DirectExactHistoryKind::sensory_contact ||
        candidate.identity != event.ticket_id || candidate.source != event.node ||
        candidate.subject != event.channel || candidate.value != event.word ||
        candidate.context != event.context ||
        candidate.resident_tick != current_tick ||
        candidate.event_tick != event.timestamp ||
        (candidate.flags & kDirectHistoryVerifiedObservation) == 0u ||
        (candidate.flags & kDirectHistoryPayloadFlags) !=
            static_cast<std::uint32_t>(CausalOrigin::external_contact))
      continue;
    sensory_record = &candidate;
    ++sensory_matches;
  }
  if (sensory_matches != 1u || sensory_record == nullptr) {
    ++frontier->refusals;
    return false;
  }
  sensory_authority.history_sequence = sensory_record->sequence;
  sensory_authority.history_content_root =
      resident_sensory_history_content_root(*sensory_record);
  sensory_authority.contact_identity = receipt->identity;
  sensory_authority.boundary_session_epoch = receipt->boundary_session_epoch;
  sensory_authority.ingress_sequence = receipt->ingress_sequence;
  sensory_authority.resident_tick = sensory_record->resident_tick;
  sensory_authority.event_tick = sensory_record->event_tick;
  sensory_authority.history_flags = sensory_record->flags;
  sensory_authority.authority_incarnation =
      resident_contact_authority_incarnation(*receipt, event);
  sensory_authority.witness_version = kResidentSensoryWitnessVersion;
  sensory_authority.claim_incarnation = participation.claim_incarnation;
  sensory_authority.receipt_content_root =
      resident_sensory_receipt_content_root(
          event.ticket_id, contact, sensory_authority);
  if (!can_record_resident_raw_contact_binding(
          state, event.ticket_id, contact, sensory_authority,
          static_cast<std::uint32_t>(CausalOrigin::external_contact))) {
    ++frontier->refusals;
    return false;
  }
  const std::uint32_t bootstrap_matches =
      resident_bootstrap_incidence_count(*state, event.node);
  std::uint32_t exact_index = kInvalidIndex, exact_matches = 0u;
  std::uint32_t wildcard_index = kInvalidIndex, wildcard_matches = 0u;
  bool malformed_exact = false;
  if (state->recipe_incidence_count <= kResidentRecipeIncidenceCapacity)
    for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i) {
      if (resident_raw_contact_key_equal(
              state->recipe_incidence[i].contact, contact)) {
        if (state->recipe_incidence[i].node != event.node ||
            state->recipe_incidence[i].contact.node != event.node)
          malformed_exact = true;
        else {
          exact_index = i;
          ++exact_matches;
        }
      } else if (state->recipe_incidence[i].node == event.node &&
                 resident_raw_contact_key_empty(
                     state->recipe_incidence[i].contact)) {
        if (wildcard_index == kInvalidIndex)
          wildcard_index = i;
        ++wildcard_matches;
      }
    }
  // After opaque surfaces exist, refuse wildcard absorption of novel Words.
  // Motor Words used as outcome-as-ground are registered as exact surfaces at
  // emit time (resident_register_motor_ground_surfaces), so they exact-match.
  const bool allow_bootstrap_wildcard =
      resident_exact_surface_incidence_count(*state, event.node) == 0u;
  const std::uint32_t incidence_matches =
      exact_matches != 0u
          ? exact_matches
          : (allow_bootstrap_wildcard ? wildcard_matches : 0u);
  const std::uint32_t incidence_index =
      exact_matches != 0u
          ? exact_index
          : (allow_bootstrap_wildcard ? wildcard_index : kInvalidIndex);
  if (malformed_exact || participation_matches != 1u ||
      bootstrap_matches > 1u ||
      incidence_matches > 1u ||
      current_tick > 0xffffffffu - lifetime_ticks) {
    if (counters != nullptr &&
        (malformed_exact || participation_matches > 1u ||
         bootstrap_matches > 1u ||
         incidence_matches > 1u))
      ++counters->ambiguity_rejects;
    ++frontier->refusals;
    return false;
  }
  // A canonical body contact can precede the first resident Recipe incidence.
  // Transfer the authenticated claim into bounded resident matter immediately
  // so transport may reclaim/reuse the ingress ring while delayed or multi-hop
  // sparse propagation continues. No Recipe occurrence is manufactured here.
  if (incidence_matches == 0u) {
    // A blank-grown brain may not yet have unfolded any route Recipe at this
    // sensor. Break the structural bootstrap cycle by resident-local
    // nomination of one existing active route. Contact content supplies no
    // parameter, support, evidence or credit; it only identifies the physical
    // node whose already-grown morphology must become executable.
    if (bootstrap_matches == 0u) {
#if defined(__CUDA_ARCH__)
      const DirectNode& source = brain.nodes[event.node];
      std::uint32_t selected_route = kInvalidIndex;
      std::uint64_t selected_abs_conductance = 0u;
      for (std::uint32_t slot = 0u; slot < source.route_capacity; ++slot) {
        const std::uint32_t route_index = source.route_offset + slot;
        if (route_index >= brain.route_capacity) break;
        const DirectRoute& route = brain.routes[route_index];
        if (!route_is_active(route) || route.source != event.node ||
            route.target >= brain.node_count ||
            brain.route_incarnations[route_index] == 0u)
          continue;
        const std::uint64_t magnitude = route.conductance_q16 < 0
            ? static_cast<std::uint64_t>(-
                  static_cast<std::int64_t>(route.conductance_q16))
            : static_cast<std::uint64_t>(route.conductance_q16);
        if (selected_route == kInvalidIndex ||
            magnitude > selected_abs_conductance ||
            (magnitude == selected_abs_conductance &&
             route_index < selected_route)) {
          selected_route = route_index;
          selected_abs_conductance = magnitude;
        }
      }
      std::uint32_t recipe_cell = kInvalidIndex;
      if (selected_route == kInvalidIndex ||
          !materialize_postbirth_route_recipe_parent_external(
              brain, brain.routes[selected_route], selected_route,
              brain.route_incarnations[selected_route], &recipe_cell) ||
          recipe_cell == kInvalidIndex) {
        ++frontier->refusals;
        return false;
      }
#else
      ++frontier->refusals;
      return false;
#endif
    }
    if ((frontier->boundary_session_epoch != 0u &&
         frontier->boundary_session_epoch != receipt->boundary_session_epoch) ||
        receipt->ingress_sequence <= frontier->last_ingress_sequence) {
      if (counters != nullptr) ++counters->stale_currentness_rejects;
      ++frontier->refusals;
      return false;
    }
    expire_resident_actual_frontier(frontier, current_tick, counters);
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
      const auto& live = frontier->entries[i];
      const auto& pending = frontier->pending_contacts[i];
      if ((live.state == ResidentActualFrontierState::live &&
           live.contact_identity == receipt->identity) ||
          (pending.state == ResidentPendingActualContactState::pending &&
           (pending.receipt.identity == receipt->identity ||
            pending.receipt.ingress_sequence == receipt->ingress_sequence))) {
        ++frontier->refusals;
        if (counters != nullptr) ++counters->stale_currentness_rejects;
        return false;
      }
    }
    std::uint32_t pending_slot = kInvalidIndex;
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
      if (pending_slot == kInvalidIndex &&
          frontier->pending_contacts[i].state !=
              ResidentPendingActualContactState::pending)
        pending_slot = i;
    if (pending_slot == kInvalidIndex) {
      ++frontier->refusals;
      return false;
    }
    if (!record_resident_raw_contact_binding(
            brain.postbirth_constructor, event.ticket_id, contact,
            sensory_authority,
            static_cast<std::uint32_t>(CausalOrigin::external_contact))) {
      ++frontier->refusals;
      return false;
    }
    ResidentPendingActualContact pending{};
    pending.event = event;
    pending.receipt = *receipt;
    pending.claim_incarnation = participation.claim_incarnation;
    pending.authority_incarnation = participation.authority_incarnation;
    pending.expiry_tick = participation.expiry_tick < current_tick + lifetime_ticks
        ? participation.expiry_tick : current_tick + lifetime_ticks;
    pending.derivation_ceiling = state->derivation_count;
    pending.state = ResidentPendingActualContactState::pending;
    frontier->pending_contacts[pending_slot] = pending;
    frontier->boundary_session_epoch = receipt->boundary_session_epoch;
    frontier->last_ingress_sequence = receipt->ingress_sequence;
    ++frontier->pending_contact_count;
    if (counters != nullptr) ++counters->deferred_contacts;
    receipt->consumed = 1u;
    ++frontier->consumed_credentials;
    return false;
  }
  const auto& incidence = state->recipe_incidence[incidence_index];
  if (incidence.derivation_index >= state->derivation_count) {
    if (counters != nullptr) ++counters->stale_currentness_rejects;
    ++frontier->refusals;
    return false;
  }
  const auto& derivation = brain.postbirth_derivations[incidence.derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
      derivation.revision_identity != incidence.revision_identity ||
      brain.recipe_cells[derivation.recipe_cell].revision_identity != incidence.revision_identity ||
      derivation.port_count == 0u ||
      derivation.source_node != incidence.node ||
      derivation.ports[0].node != incidence.node ||
      current_tick > 0xffffffffu - lifetime_ticks) {
    if (counters != nullptr) ++counters->stale_currentness_rejects;
    ++frontier->refusals;
    return false;
  }
  std::uint64_t execution_route_incarnation = incidence.route_incarnation;
  if (derivation.route_incarnations[0] != incidence.route_incarnation ||
      derivation.route_index >= brain.route_capacity ||
      !route_is_active(brain.routes[derivation.route_index]) ||
      brain.routes[derivation.route_index].source != incidence.node ||
      brain.route_incarnations[derivation.route_index] !=
          incidence.route_incarnation) {
    if (counters != nullptr) ++counters->stale_currentness_rejects;
    ++frontier->refusals;
    return false;
  }
  receipt->consumed = 1u;
  ++frontier->consumed_credentials;
  if ((frontier->boundary_session_epoch != 0u &&
       frontier->boundary_session_epoch != receipt->boundary_session_epoch) ||
      receipt->ingress_sequence <= frontier->last_ingress_sequence) {
    if (counters != nullptr) ++counters->stale_currentness_rejects;
    ++frontier->refusals;
    return false;
  }
  expire_resident_actual_frontier(frontier, current_tick, counters);
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    if (frontier->entries[i].state == ResidentActualFrontierState::live &&
        frontier->entries[i].contact_identity == receipt->identity) {
      if (counters != nullptr) ++counters->stale_currentness_rejects;
      ++frontier->refusals;
      return false;
    }
  std::uint32_t slot = kInvalidIndex;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    if (slot == kInvalidIndex && frontier->entries[i].state != ResidentActualFrontierState::live)
      slot = i;
  std::uint32_t publication_slot = kInvalidIndex;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    if (publication_slot == kInvalidIndex &&
        frontier->pending_contacts[i].state !=
            ResidentPendingActualContactState::pending &&
        frontier->pending_contacts[i].state !=
            ResidentPendingActualContactState::admitted)
      publication_slot = i;
  if (slot == kInvalidIndex || publication_slot == kInvalidIndex) {
    ++frontier->refusals;
    return false;
  }
  if (counters != nullptr) {
    counters->valid_exact_candidates += exact_matches;
    counters->valid_wildcard_candidates += wildcard_matches;
    counters->exact_claim_matches += participation_matches;
  }
  const auto& recipe = brain.recipe_cells[derivation.recipe_cell];
  std::uint32_t variables[2] = {
      resident_physical_binding_identity(
          brain, derivation.ports[0].node, event.context, event.ticket_id,
          participation.authority_incarnation),
      resident_physical_binding_identity(
          brain, derivation.ports[1].node, event.context, event.ticket_id,
          participation.authority_incarnation)};
  std::uint64_t occurrence_identity = exact_history_fold_word(
      receipt->identity, derivation.revision_identity);
  occurrence_identity = exact_history_fold_word(occurrence_identity, event.context);
  ResidentActualFrontierEntry candidate{};
  if (!bind_resident_recipe_occurrence(
          recipe, derivation, variables, 2u,
          occurrence_identity == 0u ? 1u : occurrence_identity,
          event.ticket_id, receipt->source_identity,
          participation.claim_incarnation,
          ResidentOccurrenceLineageKind::actual,
          participation.authority, event.context,
          current_tick, current_tick + lifetime_ticks, 0, &candidate.occurrence) ||
      !lower_resident_executable_morphology(
          brain, derivation, candidate.occurrence, false, &candidate.work) ||
      !execute_resident_executable_morphology(
          brain, derivation, candidate.occurrence, candidate.work,
          static_cast<std::int32_t>(event.word), &candidate.output_q16,
          &candidate.work_units)) {
    ++frontier->refusals;
    return false;
  }
  candidate.occurrence.route_incarnation = execution_route_incarnation;
  // Every executable morphology retains the exact input used by this Occurrence.
  if (!apply_resident_occurrence_bound_activation(
          &candidate.occurrence, candidate.occurrence.occurrence_identity,
          event.context, variables, 2u, current_tick,
          static_cast<std::int32_t>(event.word))) {
    ++frontier->refusals;
    return false;
  }
  candidate.contact_identity = receipt->identity;
  candidate.ingress_sequence = receipt->ingress_sequence;
  candidate.revision_identity = incidence.revision_identity;
  candidate.route_incarnation = execution_route_incarnation;
  candidate.state = ResidentActualFrontierState::live;
  candidate.derivation_index = incidence.derivation_index;
  candidate.nomination_work_units = 1u;
  candidate.composition_depth = incidence.composition_depth;
  if (!record_resident_raw_contact_binding(
          brain.postbirth_constructor, event.ticket_id, contact,
          sensory_authority,
          static_cast<std::uint32_t>(CausalOrigin::external_contact))) {
    ++frontier->refusals;
    return false;
  }
  frontier->entries[slot] = candidate;
  // Both freshly unfolded and already represented contacts traverse the same
  // resident publication mailbox. Previously only deferred first-contact
  // admissions reached sparse execution, leaving later lived contacts as
  // inert frontier entries despite valid Occurrence participation.
  ResidentPendingActualContact publication{};
  publication.event = event;
  publication.receipt = *receipt;
  publication.claim_incarnation = participation.claim_incarnation;
  publication.authority_incarnation = participation.authority_incarnation;
  publication.expiry_tick = candidate.occurrence.expiry_tick;
  publication.derivation_ceiling = state->derivation_count;
  publication.state = ResidentPendingActualContactState::admitted;
  frontier->pending_contacts[publication_slot] = publication;
  frontier->boundary_session_epoch = receipt->boundary_session_epoch;
  frontier->last_ingress_sequence = receipt->ingress_sequence;
  ++frontier->live_count;
  ++frontier->admissions;
  if ((derivation.condensation_flags &
       direct_network::kResidentDerivationCondensedNetwork) != 0u) {
    ++frontier->condensed_executions;
    if (candidate.work_units > 1u) ++frontier->condensation_deoptimizations;
  }
  return true;
}

// Resolve checkpointed pending contacts only after the executor publishes its
// next causal-participation bank. Active participation is checkpointed resident
// state; ephemeral contribution staging is deliberately not admission authority.
DIRECT_ADULT_HD inline bool admit_resident_actual_frontier_pending_participation(
    const DirectBrain& brain, const NodeCausalParticipation* active,
    std::uint32_t current_tick, ResidentActualFrontier* frontier,
    DirectActualFrontierCausalCounters* counters = nullptr) {
  using namespace direct_network;
  if (frontier == nullptr || active == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr || brain.postbirth_derivations == nullptr ||
      brain.recipe_cells == nullptr || brain.routes == nullptr ||
      brain.route_incarnations == nullptr) return false;
  const auto& state = *brain.postbirth_constructor;
  if (state.recipe_incidence_count > kResidentRecipeIncidenceCapacity ||
      state.derivation_count > state.derivation_capacity) return false;
  expire_resident_actual_frontier(frontier, current_tick, counters);
  bool changed = false;
  for (std::uint32_t ps = 0u; ps < kResidentActualFrontierCapacity; ++ps) {
    auto& pending = frontier->pending_contacts[ps];
    if (pending.state != ResidentPendingActualContactState::pending) continue;
    if (!resident_contact_credential_valid(brain, pending.event, pending.receipt,
                                           pending.receipt.ingress_sequence) ||
        pending.claim_incarnation == 0u || pending.authority_incarnation == 0u ||
        pending.derivation_ceiling > state.derivation_count) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u) --frontier->pending_contact_count;
      if (counters != nullptr) ++counters->stale_currentness_rejects;
      ++frontier->refusals; continue;
    }
    const ResidentRawContactKey contact = resident_raw_contact_key(
        pending.event.node, pending.event.channel, pending.event.word,
        pending.event.context);
    if (resident_bootstrap_incidence_count(state, pending.event.node) > 1u) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u)
        --frontier->pending_contact_count;
      ++frontier->refusals;
      if (counters != nullptr) ++counters->ambiguity_rejects;
      continue;
    }
    const ResidentRawContactKey recorded_contact =
        resident_raw_contact_binding(state, pending.event.ticket_id);
    std::uint32_t exact_binding_candidates = 0u;
    bool malformed_exact = false;
    for (std::uint32_t i = 0u; i < state.recipe_incidence_count; ++i)
      if (resident_raw_contact_key_equal(
              state.recipe_incidence[i].contact, contact)) {
        if (state.recipe_incidence[i].node != pending.event.node ||
            state.recipe_incidence[i].contact.node != pending.event.node)
          malformed_exact = true;
        else
          ++exact_binding_candidates;
      }
    if (malformed_exact || exact_binding_candidates > 1u) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u)
        --frontier->pending_contact_count;
      ++frontier->refusals;
      if (counters != nullptr) ++counters->ambiguity_rejects;
      continue;
    }
    std::uint32_t exact_selected = kInvalidIndex, exact_matches = 0u;
    std::uint32_t wildcard_selected = kInvalidIndex, wildcard_matches = 0u;
    bool saw_stale_candidate = false;
    bool malformed_exact_structure = false;
    for (std::uint32_t i = 0u; i < state.recipe_incidence_count; ++i) {
      const auto& incidence = state.recipe_incidence[i];
      const bool exact_contact =
          resident_raw_contact_key_equal(incidence.contact, contact);
      // The exact raw binding may materialize after admission; unrelated and
      // wildcard derivations remain bounded by the pre-contact ceiling.
      const bool contact_authored_after_admission =
          incidence.derivation_index >= pending.derivation_ceiling &&
          exact_contact && resident_raw_contact_key_equal(recorded_contact, contact);
      const bool candidate_contact = exact_contact ||
          (exact_binding_candidates == 0u &&
           incidence.node == pending.event.node &&
           resident_raw_contact_key_empty(incidence.contact));
      if ((!contact_authored_after_admission &&
           incidence.derivation_index >= pending.derivation_ceiling) ||
          incidence.derivation_index >= state.derivation_count ||
          incidence.node >= brain.node_count)
        continue;
      const auto& d = brain.postbirth_derivations[incidence.derivation_index];
      const bool structural_node_mismatch = exact_contact &&
          (d.source_node != incidence.node || d.port_count == 0u ||
           d.ports[0].node != incidence.node ||
           (d.route_index < brain.route_capacity &&
            brain.routes[d.route_index].source != incidence.node));
      if (structural_node_mismatch) {
        malformed_exact_structure = true;
        continue;
      }
      if (d.recipe_cell >= brain.development->recipe_cell_count ||
          d.revision_identity != incidence.revision_identity ||
          brain.recipe_cells[d.recipe_cell].revision_identity != incidence.revision_identity ||
          d.port_count == 0u || d.port_count > kResidentDerivationWidth ||
          d.source_node != incidence.node ||
          d.ports[0].node != incidence.node ||
          d.route_index >= brain.route_capacity ||
          d.route_incarnations[0] != incidence.route_incarnation ||
          !route_is_active(brain.routes[d.route_index]) ||
          brain.routes[d.route_index].source != incidence.node ||
          brain.route_incarnations[d.route_index] != incidence.route_incarnation ||
          (contact_authored_after_admission && d.witness_identity == 0u)) {
        saw_stale_candidate = saw_stale_candidate || candidate_contact;
        continue;
      }
      // Initial admission already required exactly one authenticated current
      // participation and froze its claim/authority/expiry into this resident,
      // checkpointed pending record. Do not make that evidence disappear just
      // because the small transient node bank later carries redundant fan-in
      // rows or reuses them before the Recipe derivation materializes.
      if (pending.expiry_tick < current_tick) continue;
      if (counters != nullptr) ++counters->exact_claim_matches;
      if (exact_contact) {
        if (counters != nullptr) ++counters->valid_exact_candidates;
        exact_selected = i;
        ++exact_matches;
      } else if (exact_binding_candidates == 0u &&
                 incidence.node == pending.event.node &&
                 resident_raw_contact_key_empty(incidence.contact)) {
        if (counters != nullptr) ++counters->valid_wildcard_candidates;
        if (wildcard_selected == kInvalidIndex) {
          wildcard_selected = i;
          wildcard_matches = 1u;
        } else
          ++wildcard_matches;
      }
    }
    const bool allow_bootstrap_wildcard =
        resident_exact_surface_incidence_count(state, pending.event.node) == 0u;
    const std::uint32_t matches =
        exact_matches != 0u
            ? exact_matches
            : (allow_bootstrap_wildcard ? wildcard_matches : 0u);
    const std::uint32_t selected =
        exact_matches != 0u
            ? exact_selected
            : (allow_bootstrap_wildcard ? wildcard_selected : kInvalidIndex);
    if (malformed_exact_structure) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u)
        --frontier->pending_contact_count;
      if (counters != nullptr) ++counters->stale_currentness_rejects;
      ++frontier->refusals;
      continue;
    }
    if (matches == 0u) {
      if (counters != nullptr && saw_stale_candidate)
        ++counters->stale_currentness_rejects;
      continue;
    }
    if (matches != 1u || selected == kInvalidIndex) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u) --frontier->pending_contact_count;
      if (counters != nullptr) ++counters->ambiguity_rejects;
      ++frontier->refusals; continue;
    }
    std::uint32_t free_slot = kInvalidIndex;
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
      if (free_slot == kInvalidIndex && frontier->entries[i].state != ResidentActualFrontierState::live)
        free_slot = i;
    if (free_slot == kInvalidIndex) continue;
    const auto& incidence = state.recipe_incidence[selected];
    const auto& d = brain.postbirth_derivations[incidence.derivation_index];
    const auto& recipe = brain.recipe_cells[d.recipe_cell];
    // The actual contact value is frozen evidence. Current node activation is
    // transient recurrence and may decay while resident construction catches
    // up; it cannot replace or erase the contact that actually participated.
    const std::int32_t input = static_cast<std::int32_t>(pending.event.word);
    std::uint32_t vars[kResidentDerivationWidth]{};
    for (std::uint32_t p = 0u; p < d.port_count; ++p)
      vars[p] = resident_physical_binding_identity(brain, d.ports[p].node,
          pending.event.context, pending.event.ticket_id, pending.authority_incarnation);
    std::uint64_t oid = exact_history_fold_word(pending.receipt.identity, d.revision_identity);
    oid = exact_history_fold_word(oid, pending.event.context);
    oid = exact_history_fold_word(oid, incidence.route_incarnation);
    ResidentActualFrontierEntry candidate{};
    if (!bind_resident_recipe_occurrence(recipe, d, vars, d.port_count, oid == 0u ? 1u : oid,
            pending.event.ticket_id, pending.receipt.source_identity, pending.claim_incarnation,
            ResidentOccurrenceLineageKind::actual, DirectParticipationAuthority::independent_external,
            pending.event.context, pending.event.timestamp, pending.expiry_tick, 0, &candidate.occurrence) ||
        !lower_resident_executable_morphology(brain, d, candidate.occurrence, false, &candidate.work) ||
        !execute_resident_executable_morphology(brain, d, candidate.occurrence, candidate.work,
                                                input, &candidate.output_q16, &candidate.work_units) ||
        !apply_resident_occurrence_bound_activation(
            &candidate.occurrence, candidate.occurrence.occurrence_identity,
            pending.event.context, vars, d.port_count, pending.event.timestamp, input)) {
      pending.state = ResidentPendingActualContactState::expired;
      if (frontier->pending_contact_count != 0u) --frontier->pending_contact_count;
      ++frontier->refusals;
      continue;
    }
    candidate.occurrence.route_incarnation = incidence.route_incarnation;
    candidate.contact_identity = pending.receipt.identity;
    candidate.ingress_sequence = pending.receipt.ingress_sequence;
    candidate.revision_identity = incidence.revision_identity;
    candidate.route_incarnation = incidence.route_incarnation;
    candidate.state = ResidentActualFrontierState::live;
    candidate.derivation_index = incidence.derivation_index;
    candidate.nomination_work_units = state.recipe_incidence_count;
    candidate.composition_depth = incidence.composition_depth;
    frontier->entries[free_slot] = candidate;
    ++frontier->live_count; ++frontier->admissions;
    pending.state = ResidentPendingActualContactState::admitted;
    if (counters != nullptr) ++counters->pending_admissions;
    if (frontier->pending_contact_count != 0u) --frontier->pending_contact_count;
    changed = true;
  }
  return changed;
}

// Extend one admitted actual root only through route participations that the
// resident executor actually staged for that same external claim. The pass is
// bounded by the current contribution buffer and the four-slot actual frontier;
// it never scans dormant route morphology.
__device__ inline bool admit_resident_actual_frontier_route_descendant(
    DirectBrain brain, const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count, std::uint32_t contribution_capacity,
    std::uint32_t current_tick, ResidentActualFrontier* frontier) {
  using namespace direct_network;
  if (frontier == nullptr || contributions == nullptr || contribution_count == nullptr ||
      brain.development == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      brain.routes == nullptr || brain.route_incarnations == nullptr ||
      *contribution_count > contribution_capacity ||
      frontier->live_count >= kResidentActualFrontierCapacity)
    return false;
  auto* state = brain.postbirth_constructor;
  for (std::uint32_t root_slot = 0u; root_slot < kResidentActualFrontierCapacity; ++root_slot) {
    auto& root = frontier->entries[root_slot];
    if (root.state != ResidentActualFrontierState::live ||
        root.occurrence.authority != DirectParticipationAuthority::independent_external ||
        root.derivation_index >= state->derivation_count)
      continue;
    const auto& root_d = brain.postbirth_derivations[root.derivation_index];
    if (root_d.port_count != 2u) continue;
    const bool reopened = root.work_units > 1u &&
        (root_d.condensation_flags & kResidentDerivationCondensedNetwork) != 0u &&
        root_d.condensation_source_count == kResidentCondensationSourceCount;
    if (reopened) {
      std::uint32_t source_slots[kResidentCondensationSourceCount] = {
          kInvalidIndex, kInvalidIndex};
      for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
        const auto& existing = frontier->entries[slot];
        if (slot == root_slot || existing.state != ResidentActualFrontierState::live ||
            existing.occurrence.participation_identity != root.occurrence.participation_identity ||
            existing.occurrence.source_identity != root.occurrence.source_identity ||
            existing.occurrence.source_incarnation != root.occurrence.source_incarnation ||
            existing.derivation_index >= state->derivation_count)
          continue;
        const std::uint32_t route_index =
            brain.postbirth_derivations[existing.derivation_index].route_index;
        for (std::uint32_t source = 0u; source < kResidentCondensationSourceCount; ++source)
          if (route_index == root_d.condensation_sources[source].route_index)
            source_slots[source] = slot;
      }
      const std::uint32_t source = source_slots[0] == kInvalidIndex ? 0u :
          (source_slots[1] == kInvalidIndex ? 1u : kResidentCondensationSourceCount);
      if (source == kResidentCondensationSourceCount) {
        root.state = ResidentActualFrontierState::settled;
        root.occurrence.state = kResidentRecipeOccurrenceSettled;
        if (frontier->live_count != 0u) --frontier->live_count;
        const auto& first = frontier->entries[source_slots[0]];
        record_resident_recipe_incidence(
            state, first.derivation_index,
            brain.postbirth_derivations[first.derivation_index]);
        return true;
      }
      const auto& snapshot = root_d.condensation_sources[source];
      std::uint32_t selected = kInvalidIndex;
      for (std::uint32_t i = 0u; i < *contribution_count; ++i) {
        const auto& descriptor = contributions[i];
        if (descriptor.contribution_kind == DirectContributionKind::sparse_route &&
            descriptor.authority == DirectParticipationAuthority::independent_external &&
            descriptor.ticket_id == root.occurrence.participation_identity &&
            descriptor.claim_incarnation == root.occurrence.source_incarnation &&
            descriptor.route_index == snapshot.route_index &&
            descriptor.route_incarnation == snapshot.route_incarnation &&
            descriptor.expiry_tick >= current_tick &&
            descriptor.route_index < brain.route_capacity &&
            brain.route_incarnations[descriptor.route_index] == descriptor.route_incarnation) {
          selected = i;
          break;
        }
      }
      if (selected == kInvalidIndex) continue;
      const auto& descriptor = contributions[selected];
      DirectRoute& route = brain.routes[descriptor.route_index];
      const std::uint32_t recipe_cell = decode_route_recipe_builder(route.flags);
      if (recipe_cell >= brain.development->recipe_cell_count || !route_is_active(route) ||
          route.source != descriptor.source_node || route.target != descriptor.target_node)
        return false;
      std::uint32_t derivation_index = kInvalidIndex;
      for (std::uint32_t d = 0u; d < state->derivation_count; ++d) {
        const auto& candidate = brain.postbirth_derivations[d];
        if (candidate.recipe_cell == recipe_cell &&
            candidate.route_index == descriptor.route_index &&
            candidate.generation == 0u &&
            candidate.route_incarnations[0] == descriptor.route_incarnation) {
          if (derivation_index != kInvalidIndex) return false;
          derivation_index = d;
        }
      }
      if (derivation_index == kInvalidIndex) return false;
      auto& derivation = brain.postbirth_derivations[derivation_index];
      auto& recipe = brain.recipe_cells[recipe_cell];
      if (derivation.relation_count != 1u || derivation.parameter_count != 1u ||
          derivation.relations[0] !=
              static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain))
        return false;
      if (derivation.parameters_q16[0] != route.conductance_q16) {
        std::uint64_t evidence = exact_history_fold_word(
            0x726f757465703072ull, root.occurrence.occurrence_identity);
        evidence = exact_history_fold_word(evidence, root.occurrence.source_identity);
        evidence = exact_history_fold_word(evidence, descriptor.route_index);
        evidence = exact_history_fold_word(evidence, descriptor.route_incarnation);
        evidence = exact_history_fold_word(
            evidence, static_cast<std::uint32_t>(route.conductance_q16));
        DirectExactHistoryRecord event{};
        if (!stage_resident_recipe_revision_event(
                &event, recipe, recipe_cell, ResidentRecipeRevisionAuthority::experience,
                root.occurrence.occurrence_identity, evidence == 0u ? 1u : evidence,
                current_tick, current_tick, descriptor.route_index, source, 0u,
                static_cast<std::int64_t>(route.conductance_q16) -
                    derivation.parameters_q16[0]))
          return false;
        ResidentRecipeCell predicted = recipe;
        if (!apply_resident_recipe_revision_event(&predicted, event, recipe_cell) ||
            !begin_exact_history_phase(
                &brain.development->exact_history,
                DirectExactHistoryKind::recipe_revision, 1u, current_tick))
          return false;
        brain.development->exact_history.records[
            brain.development->exact_history.phase_base] = event;
        if (finish_exact_history_phase(&brain.development->exact_history) != 1u)
          return false;
        const std::uint64_t prior_revision = recipe.revision_identity;
        recipe = predicted;
        derivation.revision_identity = recipe.revision_identity;
        derivation.parameters_q16[0] = route.conductance_q16;
        for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i) {
          auto& incidence = state->recipe_incidence[i];
          if (incidence.derivation_index == derivation_index &&
              incidence.revision_identity == prior_revision)
            incidence.revision_identity = derivation.revision_identity;
        }
      }
      std::uint32_t variables[2] = {
          resident_physical_binding_identity(
              brain, derivation.ports[0].node, root.occurrence.context_signature,
              root.occurrence.participation_identity, descriptor.authority_incarnation),
          resident_physical_binding_identity(
              brain, derivation.ports[1].node, root.occurrence.context_signature,
              root.occurrence.participation_identity, descriptor.authority_incarnation)};
      std::uint64_t occurrence_identity = exact_history_fold_word(
          root.occurrence.occurrence_identity, descriptor.route_index);
      occurrence_identity = exact_history_fold_word(
          occurrence_identity, descriptor.route_incarnation);
      occurrence_identity = exact_history_fold_word(
          occurrence_identity, 0x72656f70656e7030ull);
      ResidentActualFrontierEntry child{};
      const std::int32_t input_q16 = source == 0u
          ? root.occurrence.activation_q16
          : frontier->entries[source_slots[0]].output_q16;
      if (!bind_resident_recipe_occurrence(
              recipe, derivation, variables, 2u,
              occurrence_identity == 0u ? 1u : occurrence_identity,
              root.occurrence.participation_identity, root.occurrence.source_identity,
              root.occurrence.source_incarnation, ResidentOccurrenceLineageKind::actual,
              DirectParticipationAuthority::resident_external_descendant,
              root.occurrence.context_signature, current_tick,
              root.occurrence.expiry_tick < descriptor.expiry_tick
                  ? root.occurrence.expiry_tick : descriptor.expiry_tick,
              descriptor.frozen_eligibility_q16, &child.occurrence) ||
          !lower_resident_executable_morphology(
              brain, derivation, child.occurrence, false, &child.work) ||
          !execute_resident_executable_morphology(
              brain, derivation, child.occurrence, child.work, input_q16,
              &child.output_q16, &child.work_units) ||
          !apply_resident_occurrence_bound_activation(
              &child.occurrence, child.occurrence.occurrence_identity,
              root.occurrence.context_signature, variables, 2u, current_tick,
              input_q16))
        return false;
      child.occurrence.route_incarnation = descriptor.route_incarnation;
      child.contact_identity = exact_history_fold_word(
          root.contact_identity, descriptor.route_index);
      child.ingress_sequence = root.ingress_sequence;
      child.revision_identity = derivation.revision_identity;
      child.route_incarnation = descriptor.route_incarnation;
      child.state = ResidentActualFrontierState::live;
      child.derivation_index = derivation_index;
      child.nomination_work_units = 1u;
      std::uint32_t free_slot = kInvalidIndex;
      for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot)
        if (frontier->entries[slot].state != ResidentActualFrontierState::live) {
          free_slot = slot;
          break;
        }
      if (free_slot == kInvalidIndex) return false;
      frontier->entries[free_slot] = child;
      ++frontier->live_count;
      if (source == 1u) {
        root.state = ResidentActualFrontierState::settled;
        root.occurrence.state = kResidentRecipeOccurrenceSettled;
        --frontier->live_count;
        const auto& first = frontier->entries[source_slots[0]];
        record_resident_recipe_incidence(
            state, first.derivation_index,
            brain.postbirth_derivations[first.derivation_index]);
      }
      return true;
    }
    bool already_extended = false;
    for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
      const auto& existing = frontier->entries[slot];
      already_extended |= existing.state == ResidentActualFrontierState::live &&
          existing.occurrence.authority ==
              DirectParticipationAuthority::resident_external_descendant &&
          existing.occurrence.participation_identity == root.occurrence.participation_identity &&
          existing.occurrence.source_identity == root.occurrence.source_identity &&
          existing.occurrence.source_incarnation == root.occurrence.source_incarnation;
    }
    if (already_extended) continue;
    std::uint32_t best = kInvalidIndex;
    for (std::uint32_t i = 0u; i < *contribution_count; ++i) {
      const auto& descriptor = contributions[i];
      if (descriptor.contribution_kind != DirectContributionKind::sparse_route ||
          descriptor.authority != DirectParticipationAuthority::independent_external ||
          descriptor.ticket_id != root.occurrence.participation_identity ||
          descriptor.claim_incarnation != root.occurrence.source_incarnation ||
          descriptor.source_node != root_d.ports[1].node ||
          descriptor.route_index >= brain.route_capacity ||
          descriptor.route_incarnation == 0u ||
          descriptor.expiry_tick < current_tick)
        continue;
      const DirectRoute& route = brain.routes[descriptor.route_index];
      if (!route_is_active(route) || route.source != descriptor.source_node ||
          route.target != descriptor.target_node ||
          brain.route_incarnations[descriptor.route_index] != descriptor.route_incarnation)
        continue;
      if (best == kInvalidIndex || descriptor.route_index < contributions[best].route_index)
        best = i;
    }
    if (best == kInvalidIndex) continue;
    const auto& descriptor = contributions[best];
    DirectRoute& route = brain.routes[descriptor.route_index];
    std::uint32_t recipe_cell = kInvalidIndex;
    if (!materialize_postbirth_route_recipe_parent_external(
            brain, route, descriptor.route_index, descriptor.route_incarnation,
            &recipe_cell))
      return false;
    std::uint32_t derivation_index = kInvalidIndex;
    for (std::uint32_t d = 0u; d < state->derivation_count; ++d)
      if (brain.postbirth_derivations[d].recipe_cell == recipe_cell &&
          brain.postbirth_derivations[d].route_index == descriptor.route_index) {
        derivation_index = d;
        break;
      }
    if (derivation_index == kInvalidIndex) return false;
    const auto& derivation = brain.postbirth_derivations[derivation_index];
    const auto& recipe = brain.recipe_cells[recipe_cell];
    std::uint32_t variables[2] = {
        resident_physical_binding_identity(
            brain, derivation.ports[0].node, root.occurrence.context_signature,
            root.occurrence.participation_identity, descriptor.authority_incarnation),
        resident_physical_binding_identity(
            brain, derivation.ports[1].node, root.occurrence.context_signature,
            root.occurrence.participation_identity, descriptor.authority_incarnation)};
    std::uint64_t occurrence_identity = exact_history_fold_word(
        root.occurrence.occurrence_identity, descriptor.route_index);
    occurrence_identity = exact_history_fold_word(
        occurrence_identity, descriptor.route_incarnation);
    ResidentActualFrontierEntry descendant{};
    if (!bind_resident_recipe_occurrence(
            recipe, derivation, variables, 2u,
            occurrence_identity == 0u ? 1u : occurrence_identity,
            root.occurrence.participation_identity, root.occurrence.source_identity,
            root.occurrence.source_incarnation, ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::resident_external_descendant,
            root.occurrence.context_signature, current_tick,
            root.occurrence.expiry_tick < descriptor.expiry_tick
                ? root.occurrence.expiry_tick : descriptor.expiry_tick,
            descriptor.frozen_eligibility_q16, &descendant.occurrence) ||
        !lower_resident_executable_morphology(
            brain, derivation, descendant.occurrence, false, &descendant.work) ||
        !execute_resident_executable_morphology(
            brain, derivation, descendant.occurrence, descendant.work,
            root.output_q16, &descendant.output_q16, &descendant.work_units))
      return false;
    descendant.occurrence.route_incarnation = descriptor.route_incarnation;
    descendant.contact_identity = exact_history_fold_word(
        root.contact_identity, descriptor.route_index);
    descendant.ingress_sequence = root.ingress_sequence;
    descendant.revision_identity = derivation.revision_identity;
    descendant.route_incarnation = descriptor.route_incarnation;
    descendant.state = ResidentActualFrontierState::live;
    descendant.derivation_index = derivation_index;
    descendant.nomination_work_units = 1u;
    for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot)
      if (frontier->entries[slot].state != ResidentActualFrontierState::live) {
        frontier->entries[slot] = descendant;
        ++frontier->live_count;
        return true;
      }
    return false;
  }
  return false;
}

// Consume only the bounded, runtime-owned actual frontier. Two independently
// lived contexts of the same physically coupled recipe pair are sufficient to
// nominate one ordinary higher-rank revision; no host candidate or morphology
// scan participates.
DIRECT_ADULT_HD inline bool resident_actual_frontier_entry_current(
    const DirectBrain& brain,
    const direct_network::ResidentPostbirthConstructorState& state,
    const ResidentActualFrontierEntry& entry) {
  using namespace direct_network;
  if (entry.state != ResidentActualFrontierState::live ||
      entry.derivation_index >= state.derivation_count ||
      entry.occurrence.lineage_kind != ResidentOccurrenceLineageKind::actual ||
      entry.occurrence.authority == DirectParticipationAuthority::none ||
      entry.occurrence.source_identity == 0u ||
      entry.occurrence.source_incarnation == 0u ||
      entry.occurrence.route_incarnation == 0u ||
      entry.route_incarnation != entry.occurrence.route_incarnation)
    return false;
  const auto& derivation = brain.postbirth_derivations[entry.derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
      derivation.route_index >= brain.route_capacity ||
      derivation.logical_recipe_id != entry.occurrence.logical_recipe_id ||
      derivation.revision_identity != entry.revision_identity ||
      derivation.revision_identity != entry.occurrence.revision_identity ||
      derivation.route_incarnations[0] != entry.route_incarnation ||
      brain.recipe_cells[derivation.recipe_cell].logical_recipe_id !=
          derivation.logical_recipe_id ||
      brain.recipe_cells[derivation.recipe_cell].revision_identity !=
          derivation.revision_identity ||
      !route_is_active(brain.routes[derivation.route_index]) ||
      brain.route_incarnations[derivation.route_index] != entry.route_incarnation)
    return false;
  return true;
}

// Materialize the causal edge already executed by one current actual
// Occurrence. Eligibility admission and target mutation remain executor-owned.
DIRECT_ADULT_HD inline bool resident_actual_occurrence_contribution(
    const DirectBrain& brain, const ResidentActualFrontier& frontier,
    std::uint64_t contact_identity, std::uint64_t ingress_sequence,
    std::uint32_t authority_incarnation, std::uint32_t current_tick,
    DirectParticipationDescriptor* descriptor, std::int32_t* drive_q16) {
  using namespace direct_network;
  if (descriptor == nullptr || drive_q16 == nullptr ||
      brain.postbirth_constructor == nullptr || brain.postbirth_derivations == nullptr ||
      brain.routes == nullptr || brain.route_incarnations == nullptr ||
      frontier.live_count > kResidentActualFrontierCapacity)
    return false;
  const ResidentActualFrontierEntry* match = nullptr;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const auto& entry = frontier.entries[i];
    if (entry.contact_identity != contact_identity ||
        entry.ingress_sequence != ingress_sequence ||
        entry.occurrence.expiry_tick < current_tick ||
        !resident_actual_frontier_entry_current(
            brain, *brain.postbirth_constructor, entry))
      continue;
    if (match != nullptr) return false;
    match = &entry;
  }
  if (match == nullptr || match->output_q16 == 0 || authority_incarnation == 0u)
    return false;
  const auto& derivation = brain.postbirth_derivations[match->derivation_index];
  if (derivation.port_count != 2u || derivation.route_index >= brain.route_capacity ||
      derivation.recipe_cell >= brain.development->recipe_cell_count)
    return false;
  const DirectRoute& route = brain.routes[derivation.route_index];
  if (!route_is_active(route) || route.source != derivation.ports[0].node ||
      route.target != derivation.ports[1].node || route.target >= brain.node_count ||
      brain.route_incarnations[derivation.route_index] != match->route_incarnation)
    return false;
  DirectParticipationDescriptor value{};
  value.ticket_id = match->occurrence.participation_identity;
  value.source_node = route.source;
  value.target_node = route.target;
  value.route_index = derivation.route_index;
  value.context_signature =
      route.eligibility_context ^ (route.source * 2654435761U);
  value.expiry_tick = match->occurrence.expiry_tick;
  value.claim_incarnation = match->occurrence.source_incarnation;
  value.route_incarnation = match->route_incarnation;
  value.authority = match->occurrence.authority;
  value.authority_incarnation = authority_incarnation;
  value.contribution_kind = DirectContributionKind::sparse_route;
  value.eligibility_slot = kInvalidIndex;
  value.parent_eligibility_ref = 0u;
  value.lineage_expiry_tick = match->occurrence.expiry_tick;
  value.ancestry_depth = 1u;
  *descriptor = value;
  *drive_q16 = match->output_q16;
  return true;
}

DIRECT_ADULT_HD inline bool resident_actual_occurrence_participation_capacity(
    const NodeCausalParticipation* participation, std::uint32_t node,
    const DirectParticipationDescriptor& descriptor,
    std::uint32_t current_tick) {
  if (participation == nullptr) return false;
  const auto* base = participation + node * kNodeParticipationAperture;
  bool reusable = false;
  for (std::uint32_t slot = 0u; slot < kNodeParticipationAperture; ++slot) {
    const NodeCausalParticipation value = base[slot];
    if (value.ticket_id == descriptor.ticket_id &&
        value.claim_incarnation == descriptor.claim_incarnation)
      return value.expiry_tick >= current_tick &&
          value.authority == descriptor.authority &&
          value.authority_incarnation == descriptor.authority_incarnation;
    reusable |= value.ticket_id == 0u || value.expiry_tick < current_tick;
  }
  return reusable;
}

__device__ inline bool collect_resident_actual_frontier_relational_network(
    const DirectBrain& brain, const ResidentActualFrontier& frontier,
    direct_network::ResidentRecipeCell* recipes,
    direct_network::ResidentRecipeDerivation* derivations,
    ResidentRecipeOccurrence* occurrences,
    ResidentOccurrenceCoupling* couplings, std::uint32_t* occurrence_count,
    std::uint32_t* coupling_count) {
  using namespace direct_network;
  if (recipes == nullptr || derivations == nullptr || occurrences == nullptr ||
      couplings == nullptr || occurrence_count == nullptr ||
      coupling_count == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      frontier.live_count > kResidentActualFrontierCapacity)
    return false;
  *occurrence_count = 0u;
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
    const auto& entry = frontier.entries[slot];
    if (entry.state != ResidentActualFrontierState::live) continue;
    if (*occurrence_count >= kResidentRelationalNetworkMaxOccurrences ||
        !resident_actual_frontier_entry_current(
            brain, *brain.postbirth_constructor, entry))
      return false;
    const auto& derivation = brain.postbirth_derivations[entry.derivation_index];
    recipes[*occurrence_count] = brain.recipe_cells[derivation.recipe_cell];
    derivations[*occurrence_count] = derivation;
    occurrences[*occurrence_count] = entry.occurrence;
    ++*occurrence_count;
  }
  if (*occurrence_count != frontier.live_count) return false;
  *coupling_count = 0u;
  for (std::uint32_t source = 0u; source < *occurrence_count; ++source)
    for (std::uint32_t target = 0u; target < *occurrence_count; ++target) {
      if (source == target) continue;
      for (std::uint16_t source_port = 0u;
           source_port < derivations[source].port_count; ++source_port)
        for (std::uint16_t target_port = 0u;
             target_port < derivations[target].port_count; ++target_port) {
          ResidentOccurrenceCoupling coupling{};
          if (!bind_resident_occurrence_coupling(
                  occurrences[source], derivations[source], source_port,
                  occurrences[target], derivations[target], target_port,
                  &coupling))
            continue;
          if (*coupling_count >= kResidentRelationalNetworkMaxCouplings)
            return false;
          couplings[(*coupling_count)++] = coupling;
        }
    }
  return true;
}

// Materialize every connected Network from the bounded actual frontier.
// Disconnected coalitions become distinct coactive Networks; isolates stay
// live without singleton Network minting. Malformed Network-sized components
// refuse the whole publication atomically. No host member list.
static_assert(kResidentActualFrontierCapacity >=
                  kResidentRelationalNetworkMaxOccurrences,
              "#1641 frontier must hold a full Network Occurrence bound");
__device__ inline bool observe_resident_actual_frontier_coactive_relational_networks(
    const DirectBrain& brain, const ResidentActualFrontier& frontier,
    ResidentRelationalNetworkClosure* outs, std::uint32_t max_outs,
    std::uint32_t* out_count) {
  using namespace direct_network;
  if (outs == nullptr || out_count == nullptr || max_outs == 0u) return false;
  *out_count = 0u;
  // Shared scratch keeps Recipe/Occurrence/Coupling working sets off the
  // default CUDA thread stack. Observe is single-flight within a block.
  __shared__ ResidentRecipeCell recipes
      [kResidentRelationalNetworkMaxOccurrences];
  __shared__ ResidentRecipeDerivation derivations
      [kResidentRelationalNetworkMaxOccurrences];
  __shared__ ResidentRecipeOccurrence occurrences
      [kResidentRelationalNetworkMaxOccurrences];
  __shared__ ResidentOccurrenceCoupling couplings
      [kResidentRelationalNetworkMaxCouplings];
  for (std::uint32_t i = 0u; i < kResidentRelationalNetworkMaxOccurrences; ++i) {
    recipes[i] = ResidentRecipeCell{};
    derivations[i] = ResidentRecipeDerivation{};
    occurrences[i] = ResidentRecipeOccurrence{};
  }
  for (std::uint32_t i = 0u; i < kResidentRelationalNetworkMaxCouplings; ++i)
    couplings[i] = ResidentOccurrenceCoupling{};
  std::uint32_t occurrence_count = 0u, coupling_count = 0u;
  if (!collect_resident_actual_frontier_relational_network(
          brain, frontier, recipes, derivations, occurrences, couplings,
          &occurrence_count, &coupling_count))
    return false;
  return bind_resident_coactive_relational_network_closures(
      recipes, derivations, occurrences, occurrence_count, couplings,
      coupling_count, outs, max_outs, out_count);
}

// Convenience: succeed only when the frontier yields exactly one Network.
// Prefer observe_resident_actual_frontier_coactive_relational_networks when
// coactive coalitions are in scope (#1641).
__device__ inline bool observe_resident_actual_frontier_relational_network(
    const DirectBrain& brain, const ResidentActualFrontier& frontier,
    ResidentRelationalNetworkClosure* out) {
  if (out == nullptr) return false;
  // Stage into a temporary so a multi-Network frontier refuses without
  // mutating *out (byte-atomic publish for the single-Network API).
  ResidentRelationalNetworkClosure staged{};
  std::uint32_t count = 0u;
  if (!observe_resident_actual_frontier_coactive_relational_networks(
          brain, frontier, &staged, 1u, &count) ||
      count != 1u)
    return false;
  *out = staged;
  return true;
}
#include "hardware_native/direct_adult_action_network_unfolding.cuh"
__device__ inline bool compose_resident_actual_frontier_network_boundary(
    const DirectBrain& brain, const ResidentActualFrontier& frontier,
    ResidentRelationalNetworkClosure* closure_out,
    ResidentNetworkBoundaryRelation* relation_out) {
  using namespace direct_network;
  if (closure_out == nullptr || relation_out == nullptr) return false;
  ResidentRecipeCell recipes[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentRecipeDerivation derivations[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentRecipeOccurrence occurrences[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentOccurrenceCoupling couplings[kResidentRelationalNetworkMaxCouplings]{};
  std::uint32_t occurrence_count = 0u, coupling_count = 0u;
  ResidentRelationalNetworkClosure closure{};
  ResidentNetworkBoundaryRelation relation{};
  if (!collect_resident_actual_frontier_relational_network(
          brain, frontier, recipes, derivations, occurrences, couplings,
          &occurrence_count, &coupling_count) ||
      !bind_resident_relational_network_closure(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, &closure) ||
      !compose_resident_network_boundary_relation(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, closure, &relation))
    return false;
  *closure_out = closure;
  *relation_out = relation;
  return true;
}

__device__ inline bool condense_resident_actual_frontier_network(
    DirectBrain brain, ResidentActualFrontier* frontier) {
  using namespace direct_network;
  using substrate::direct_adult::DirectResourcePoolKind;
  if (frontier == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      brain.resource_ecology == nullptr)
    return false;
  ++frontier->condensation_nominations;
  auto* state = brain.postbirth_constructor;

  std::uint32_t left_index[2] = {kInvalidIndex, kInvalidIndex};
  std::uint32_t right_index[2] = {kInvalidIndex, kInvalidIndex};
  ResidentOccurrenceCoupling couplings[2]{};
  std::uint32_t left_derivation = kInvalidIndex;
  std::uint32_t right_derivation = kInvalidIndex;
  std::uint32_t repetitions = 0u;
  for (std::uint32_t i = 0u;
       i < kResidentActualFrontierCapacity && repetitions < 2u; ++i) {
    const auto& left = frontier->entries[i];
    if (!resident_actual_frontier_entry_current(brain, *state, left))
      continue;
    const auto& left_d = brain.postbirth_derivations[left.derivation_index];
    if (left_d.relation_count != 1u || left_d.parameter_count != 1u ||
        left_d.port_count != 2u)
      continue;
    for (std::uint32_t j = 0u;
         j < kResidentActualFrontierCapacity && repetitions < 2u; ++j) {
      const auto& right = frontier->entries[j];
      if (i == j || !resident_actual_frontier_entry_current(brain, *state, right) ||
          left.occurrence.context_signature !=
              right.occurrence.context_signature ||
          left.occurrence.participation_identity !=
              right.occurrence.participation_identity)
        continue;
      const auto& right_d = brain.postbirth_derivations[right.derivation_index];
      if (right_d.relation_count != 1u || right_d.parameter_count != 1u ||
          right_d.port_count != 2u)
        continue;
      ResidentOccurrenceCoupling coupling{};
      if (!bind_resident_occurrence_coupling(
              left.occurrence, left_d, 1u, right.occurrence, right_d, 0u,
              &coupling) || coupling.source_identity == 0u ||
          coupling.target_identity == 0u ||
          coupling.source_route_incarnation != left.route_incarnation ||
          coupling.target_route_incarnation != right.route_incarnation)
        continue;
      if (repetitions == 0u) {
        left_derivation = left.derivation_index;
        right_derivation = right.derivation_index;
      } else if (i == left_index[0] || i == right_index[0] ||
                 j == left_index[0] || j == right_index[0] ||
                 left.derivation_index != left_derivation ||
                 right.derivation_index != right_derivation ||
                 left.occurrence.context_signature ==
                     frontier->entries[left_index[0]].occurrence.context_signature) {
        continue;
      }
      left_index[repetitions] = i;
      right_index[repetitions] = j;
      couplings[repetitions] = coupling;
      ++repetitions;
    }
  }
  if (repetitions != 2u || left_derivation == right_derivation)
    return false;
  const ResidentRecipeDerivation derivations[2] = {
      brain.postbirth_derivations[left_derivation],
      brain.postbirth_derivations[right_derivation]};
  if (derivations[0].recipe_cell >= brain.development->recipe_cell_count ||
      derivations[1].recipe_cell >= brain.development->recipe_cell_count)
    return false;
  const ResidentRecipeCell recipes[2] = {
      brain.recipe_cells[derivations[0].recipe_cell],
      brain.recipe_cells[derivations[1].recipe_cell]};
  const ResidentRecipeOccurrence left_occurrences[2] = {
      frontier->entries[left_index[0]].occurrence,
      frontier->entries[left_index[1]].occurrence};
  const ResidentRecipeOccurrence right_occurrences[2] = {
      frontier->entries[right_index[0]].occurrence,
      frontier->entries[right_index[1]].occurrence};
  ResidentNetworkCondensationEvidence evidence{};
  if (!observe_resident_network_condensation(
          recipes, derivations, left_occurrences, right_occurrences, couplings,
          &evidence))
    return false;

  auto* recipe_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_record);
  auto* parent_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge);
  std::uint64_t logical = 0u, revision = 0u;
  const std::int64_t condensed_parameter =
      evidence.sources[0].relation ==
              static_cast<std::uint32_t>(ResidentRecipeRelation::trigger)
          ? static_cast<std::int64_t>(evidence.sources[0].parameter_q16) +
                evidence.sources[1].parameter_q16
          : (static_cast<std::int64_t>(evidence.sources[0].parameter_q16) *
                evidence.sources[1].parameter_q16) >> 16;
  if (recipe_pool == nullptr || parent_pool == nullptr ||
      recipe_pool->reserved_units == 0u ||
      parent_pool->reserved_units < kResidentCondensationSourceCount ||
      brain.development->recipe_cell_count >= brain.recipe_cell_capacity ||
      brain.development->recipe_cell_count >= state->recipe_cell_capacity ||
      state->derivation_count >= state->derivation_capacity ||
      state->ports_used + 2u > state->port_capacity ||
      state->relations_used + 1u > state->relation_capacity ||
      state->parameters_used + 1u > state->parameter_capacity ||
      condensed_parameter < -0x80000000ll ||
      condensed_parameter > 0x7fffffffll ||
      derivations[0].ports[0].node >= brain.node_count ||
      !replay_resident_condensation_witness(
          brain.recipe_cells, brain.development->recipe_cell_count,
          brain.postbirth_derivations, state->derivation_count, evidence,
          &logical, &revision))
    return false;
  for (std::uint32_t i = 0u; i < brain.development->recipe_cell_count; ++i)
    if (brain.recipe_cells[i].logical_recipe_id == logical) return false;
  const auto recipe_pool_before = *recipe_pool;
  const auto parent_pool_before = *parent_pool;
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u))
    return false;
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge,
          kResidentCondensationSourceCount)) {
    *recipe_pool = recipe_pool_before;
    *parent_pool = parent_pool_before;
    return false;
  }
  const std::uint16_t territory =
      brain.nodes[derivations[0].ports[0].node].territory_index;
  if (!condense_resident_recipe_network(
          brain.recipe_cells, &brain.development->recipe_cell_count,
          brain.recipe_cell_capacity, brain.postbirth_derivations, state,
          evidence, territory)) {
    *recipe_pool = recipe_pool_before;
    *parent_pool = parent_pool_before;
    return false;
  }
  for (std::uint32_t repetition = 0u; repetition < 2u; ++repetition) {
    auto& left = frontier->entries[left_index[repetition]];
    auto& right = frontier->entries[right_index[repetition]];
    left.state = right.state = ResidentActualFrontierState::settled;
    left.occurrence.state = right.occurrence.state =
        kResidentRecipeOccurrenceSettled;
  }
  std::uint32_t live = 0u;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    live += frontier->entries[i].state == ResidentActualFrontierState::live
        ? 1u : 0u;
  frontier->live_count = live;
  ++frontier->condensation_promotions;
  return true;
}

// Freeze the exact actual Occurrence behind a participation claim. The derived
// activation plane narrows candidate lanes when current; every surviving
// candidate is still verified against canonical AoS storage, and a stale or
// absent plane falls back to the full canonical scan.
DIRECT_ADULT_HD inline bool freeze_actual_occurrence_identity(
    const DirectParticipationDescriptor& participant,
    const ResidentActualFrontier* frontier,
    DirectActionParticipationLink* link,
    const ResidentActivationSoaPlane* activation_plane = nullptr) {
  if (frontier == nullptr || link == nullptr ||
      frontier->live_count > kResidentActualFrontierCapacity)
    return false;
  const bool packed =
      activation_plane != nullptr &&
      resident_activation_soa_current(*frontier, activation_plane->control,
                                      activation_plane->soa);
  const ResidentOccurrenceActivationSoa* soa =
      packed ? &activation_plane->soa : nullptr;
  const ResidentActualFrontierEntry* match = nullptr;
  const std::uint32_t candidates =
      packed ? soa->count : kResidentActualFrontierCapacity;
  for (std::uint32_t i = 0u; i < candidates; ++i) {
    const ResidentActualFrontierEntry& entry =
        frontier->entries[packed ? soa->source_slot[i] : i];
    const ResidentRecipeOccurrence& occurrence = entry.occurrence;
    if (entry.state != ResidentActualFrontierState::live ||
        occurrence.state != kResidentRecipeOccurrenceLive ||
        occurrence.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        occurrence.occurrence_identity == 0u ||
        occurrence.logical_recipe_id == 0u ||
        occurrence.revision_identity == 0u ||
        occurrence.participation_identity != participant.ticket_id ||
        occurrence.source_incarnation != participant.claim_incarnation ||
        occurrence.authority != participant.authority)
      continue;
    if (match != nullptr) return false;
    match = &entry;
  }
  if (match == nullptr) return false;
  link->logical_recipe_id = match->occurrence.logical_recipe_id;
  link->revision_identity = match->occurrence.revision_identity;
  link->occurrence_identity = match->occurrence.occurrence_identity;
  link->participation_identity = match->occurrence.participation_identity;
  link->occurrence_route_incarnation = match->occurrence.route_incarnation;
  link->occurrence_context_signature = match->occurrence.context_signature;
  link->composition_depth = match->composition_depth;
  return true;
}

// An independently returned consequence closes only the actual Occurrence
// identities frozen into that settled action. RecipeRevisions and exact causal
// history persist; the transient working occurrence must not keep competing as
// though its consequence were still outstanding.
DIRECT_ADULT_HD inline std::uint32_t settle_action_actual_occurrences(
    ResidentActualFrontier* frontier,
    const DirectActionParticipationLink* links,
    std::uint32_t participant_offset, std::uint32_t participant_count) {
  if (frontier == nullptr || links == nullptr ||
      participant_count > kMaxActionParticipationLinks ||
      frontier->live_count > kResidentActualFrontierCapacity)
    return 0u;
  std::uint32_t settled = 0u;
  for (std::uint32_t slot = 0u;
       slot < kResidentActualFrontierCapacity; ++slot) {
    auto& entry = frontier->entries[slot];
    if (entry.state != ResidentActualFrontierState::live) continue;
    for (std::uint32_t i = 0u; i < participant_count; ++i) {
      const auto& link = links[participant_offset + i];
      if (link.occurrence_identity == 0u ||
          link.participation_identity == 0u ||
          entry.occurrence.participation_identity !=
              link.participation_identity ||
          entry.occurrence.source_incarnation != link.claim_incarnation)
        continue;
      entry.state = ResidentActualFrontierState::settled;
      entry.occurrence.state = kResidentRecipeOccurrenceSettled;
      if (frontier->live_count != 0u) --frontier->live_count;
      ++settled;
      break;
    }
  }
  return settled;
}
#endif
