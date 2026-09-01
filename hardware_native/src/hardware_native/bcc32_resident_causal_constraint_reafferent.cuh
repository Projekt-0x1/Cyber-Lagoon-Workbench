#pragma once

#include <cstdint>

#include "bcc32_resident_causal_constraint_participation.cuh"
#include "bcc32_resident_causal_relation_source_witness.cuh"
#include "bcc32_resident_predictive_shadow_assay.cuh"

#if defined(__CUDACC__)
#define BCC32_PARTICIPATION_REAFFERENT_HD __host__ __device__
#else
#define BCC32_PARTICIPATION_REAFFERENT_HD
#endif

namespace substrate::bcc32::resident_causal_constraint_participation::
reafferent {

namespace participation =
    substrate::bcc32::resident_causal_constraint_participation;
namespace rewrite = substrate::bcc32::causal_rewrite;
namespace source_witness = rewrite::resident_causal_relation_source_witness;
namespace predictive_shadow =
    substrate::bcc32::resident_predictive_shadow_assay;

struct Receipt {
  std::uint32_t staged_source_found = 0u;
  std::uint32_t provenance_exact = 0u;
  std::uint32_t cursor_present = 0u;
  std::uint32_t cursor_provider_exact = 0u;
  std::uint32_t provider_locus = rewrite::kInvalid;
  std::uint32_t predicted_event = 0u;
  std::uint32_t returned_event = 0u;
  std::uint32_t matched_return = 0u;
  std::uint32_t countered_records = 0u;
  std::uint32_t admitted_records = 0u;
  std::uint32_t component_ready = 0u;
  std::uint32_t component_ambiguous = 0u;
  std::uint32_t component_records = 0u;
  std::uint32_t component_sources = 0u;
  std::uint32_t generation_closure_rederived = 0u;
  std::uint32_t rederived_event = 0u;
  std::uint32_t matched_window_start = rewrite::kInvalid;
  std::uint32_t matched_window_extent = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t source_revision = rewrite::kInvalid;
  std::uint64_t resident_revision = 0u;
  // Raw aftermath rails from the shared predictive chemistry. These are
  // receipts, never admission authority or a semantic prediction object.
  std::uint32_t omission = 0u;
  std::uint32_t unresolved_residue = 0u;
  std::uint32_t local_reaction = 0u;
  std::uint32_t predecessor_context = 0u;
  std::uint32_t predecessor_antecedent = 0u;
  std::uint32_t predecessor_recontact = 0u;

  BCC32_PARTICIPATION_REAFFERENT_HD bool admission_authorized() const {
    return staged_source_found != 0u && provenance_exact != 0u &&
           generation_closure_rederived != 0u &&
           rejected == 0u;
  }
};

namespace detail {

// This is a physical provenance window, not a semantic cell. The smallest
// causal receipt this adapter can currently prove is two external
// observations, one resident-generated/distributed event, and one later
// external consequence. A resident may append more generated/distributed
// continuation words before that consequence; those words remain part of the
// same bounded chronology rather than being mistaken for a new external source.
struct MixedProvenanceWindow {
  std::uint32_t found = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t start = rewrite::kInvalid;
  std::uint32_t extent = 0u;
  std::uint32_t generated_index = rewrite::kInvalid;
  std::uint32_t value[4]{};
  std::uint32_t generated_producer = rewrite::kInvalid;
  std::uint32_t distributed_generation = 0u;
  std::uint32_t omission = 0u;
  // Observer-only reason for a failed anchored provenance-window admission.
  // This never participates in authority, source selection, or mutation.
  std::uint32_t failure_code = 0u;
};

// A prelinguistic babble has no learned Program provider. Its exact provider
// is the unique resident motor locus plus the generic resident law over the
// two external distinctions immediately preceding the action. The body key
// and returned consequence are not available to this re-derivation.
BCC32_PARTICIPATION_REAFFERENT_HD inline bool motor_babble_provider_exact(
    const rewrite::ResidentRewriteState* state,
    const MixedProvenanceWindow& window) {
  if (state == nullptr || state->program_rules != 0u ||
      state->span_program_rules != 0u || window.found == 0u ||
      window.extent != 4u || window.generated_index != window.start + 2u ||
      window.generated_producer == rewrite::kInvalid ||
      window.generated_producer >= rewrite::live_record_capacity(state))
    return false;
  std::uint32_t expected = 0u;
  return rewrite::resident_motor_babble_action_word(
             state, window.generated_producer, window.value[0],
             window.value[1], &expected) &&
      expected == window.value[2];
}

// Ordinary fixed Programs do not carry a SpanExecutionCursor. When the
// accepted action was emitted by one, recover that provider only from its
// complete resident shape and the exact physical window. This is deliberately
// stricter than a suffix match: every possible start must be considered and
// exactly one authoritative Program must end at the generated word.
BCC32_PARTICIPATION_REAFFERENT_HD inline bool fixed_program_provider_exact(
    const rewrite::ResidentRewriteState* state,
    const rewrite::Record& trajectory,
    const MixedProvenanceWindow& window) {
  if (state == nullptr || window.found == 0u ||
      window.generated_producer == rewrite::kInvalid ||
      window.generated_producer >= rewrite::live_record_capacity(state) ||
      trajectory.lane[2] == 0u || window.start == rewrite::kInvalid ||
      window.generated_index == rewrite::kInvalid ||
      window.generated_index >= trajectory.lane[2])
    return false;
  const rewrite::Record& generated =
      state->records[window.generated_producer];
  if (generated.matter_q8 == 0u ||
      generated.lane[0] != rewrite::kFormProgram ||
      generated.lane[1] == 0u || generated.lane[2] == 0u ||
      !rewrite::resident_program_authoritative(
          state, window.generated_producer))
    return false;
  std::uint32_t matches = 0u;
  for (std::uint32_t start = 0u; start <= trajectory.lane[2]; ++start) {
    if (generated.lane[2] > trajectory.lane[2] - start ||
        !rewrite::full_program_match_at(state, generated, trajectory, start))
      continue;
    if (start + generated.lane[2] - 1u != window.generated_index)
      continue;
    ++matches;
  }
  return matches == 1u;
}

// lane[5] is the legacy first-generated offset.  A resident relation closure
// may be emitted later on the same yielded trajectory, after an ordinary
// generated term has already occupied that slot.  The exact closure anchor is
// therefore carried as a chronology-bound resident receipt and is usable only
// when the live trajectory still identifies that owner/index as distributed.
BCC32_PARTICIPATION_REAFFERENT_HD inline std::uint32_t
distributed_closure_anchor_index(const rewrite::ResidentRewriteState* state,
                                 const rewrite::Record& trajectory) {
  // The action ticket is anchored to the first generated term.  A later
  // distributed relation term may be emitted on the same trajectory after
  // the motor action (the common babble -> relation-closure sequence); using
  // that later term as the external-prefix boundary would place an earlier
  // generated word inside the required external,external,generated window.
  // Keep the legacy anchor for that chronology.  The exact receipt fields are
  // still retained for future trajectories whose first generated term is the
  // distributed closure itself.
  if (state != nullptr && trajectory.lane[1] != rewrite::kInvalid &&
      state->causal_relation_trajectory_owner == trajectory.lane[1] &&
      state->causal_relation_generated_index == trajectory.lane[5]) {
    rewrite::mixed_provenance::Origin origin{};
    std::uint32_t producer = rewrite::kInvalid;
    if (rewrite::mixed_provenance::origin_at(
            state, trajectory, state->causal_relation_generated_index, &origin,
            &producer) &&
        origin == rewrite::mixed_provenance::Origin::distributed &&
        producer == rewrite::kInvalid)
      return state->causal_relation_generated_index;
  }
  return trajectory.lane[5];
}

BCC32_PARTICIPATION_REAFFERENT_HD inline MixedProvenanceWindow
find_mixed_provenance_window(const rewrite::ResidentRewriteState* state,
                             const rewrite::Record& trajectory) {
  MixedProvenanceWindow result{};
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != rewrite::kFormTrajectory ||
      trajectory.lane[1] == 0u || trajectory.lane[1] == rewrite::kInvalid ||
      trajectory.lane[2] < 3u) {
    result.failure_code = 0xA0u;
    return result;
  }

  // The resident-generated boundary is already recorded in lane[5] and is
  // the same boundary used by distributed closure rederivation. Anchor the
  // causal receipt there: scanning arbitrary history would let a host-shaped
  // suffix choose a convenient matching window and would reintroduce the
  // ambiguity this adapter is meant to reject.
  const std::uint32_t generated_index =
      distributed_closure_anchor_index(state, trajectory);
  if (generated_index < 2u || generated_index >= trajectory.lane[2]) {
    result.failure_code = 0xA1u;
    return result;
  }
  const std::uint32_t start = generated_index - 2u;
  const bool omission_after_generated = start + 3u >= trajectory.lane[2];
  rewrite::mixed_provenance::Origin origin0{};
  rewrite::mixed_provenance::Origin origin1{};
  std::uint32_t producer0 = rewrite::kInvalid;
  std::uint32_t producer1 = rewrite::kInvalid;
  if (!rewrite::mixed_provenance::origin_at(
          state, trajectory, start, &origin0, &producer0) ||
      !rewrite::mixed_provenance::origin_at(
          state, trajectory, start + 1u, &origin1, &producer1) ||
      origin0 != rewrite::mixed_provenance::Origin::external ||
      origin1 != rewrite::mixed_provenance::Origin::external ||
      producer0 != rewrite::kInvalid || producer1 != rewrite::kInvalid) {
    result.failure_code = 0xA3u;
    return result;
  }
  std::uint32_t first_value = 0u;
  std::uint32_t second_value = 0u;
  if (!rewrite::trajectory_word_at(
          state, trajectory.lane[1], start, &first_value) ||
      !rewrite::trajectory_word_at(
          state, trajectory.lane[1], start + 1u, &second_value) ||
      first_value == 0u || first_value == rewrite::kInvalid ||
      second_value == 0u || second_value == rewrite::kInvalid) {
    result.failure_code = 0xA4u;
    return result;
  }
  rewrite::mixed_provenance::Origin generated_origin{};
  std::uint32_t generated_producer = rewrite::kInvalid;
  std::uint32_t generated_value = 0u;
  if (!rewrite::mixed_provenance::origin_at(
          state, trajectory, generated_index, &generated_origin,
          &generated_producer) ||
      (generated_origin != rewrite::mixed_provenance::Origin::generated &&
       generated_origin != rewrite::mixed_provenance::Origin::distributed) ||
      (generated_origin == rewrite::mixed_provenance::Origin::generated &&
       generated_producer == rewrite::kInvalid) ||
      (generated_origin == rewrite::mixed_provenance::Origin::distributed &&
       generated_producer != rewrite::kInvalid)) {
    result.failure_code = 0xA5u;
    return result;
  }
  if (!rewrite::trajectory_word_at(
          state, trajectory.lane[1], generated_index, &generated_value) ||
      generated_value == 0u || generated_value == rewrite::kInvalid) {
    result.failure_code = 0xA6u;
    return result;
  }
  const std::uint32_t distributed_generation =
      generated_origin == rewrite::mixed_provenance::Origin::distributed ? 1u
                                                                           : 0u;
  if (omission_after_generated) {
    result.start = start;
    result.extent = generated_index - start + 1u;
    result.generated_index = generated_index;
    result.generated_producer = generated_producer;
    result.distributed_generation = distributed_generation;
    result.value[0] = first_value;
    result.value[1] = second_value;
    result.value[2] = generated_value;
    result.omission = 1u;
    result.failure_code = 0xAEu;
    return result;
  }
  std::uint32_t end = generated_index + 1u;
  for (; end < trajectory.lane[2]; ++end) {
    rewrite::mixed_provenance::Origin origin{};
    std::uint32_t producer = rewrite::kInvalid;
    std::uint32_t value = 0u;
    if (!rewrite::mixed_provenance::origin_at(
            state, trajectory, end, &origin, &producer) ||
        !rewrite::trajectory_word_at(
            state, trajectory.lane[1], end, &value) ||
        value == 0u || value == rewrite::kInvalid)
      {
        result.failure_code = 0xA6u;
        return result;
      }
    if (origin == rewrite::mixed_provenance::Origin::generated ||
        origin == rewrite::mixed_provenance::Origin::distributed) {
      if ((origin == rewrite::mixed_provenance::Origin::generated &&
           producer == rewrite::kInvalid) ||
          (origin == rewrite::mixed_provenance::Origin::distributed &&
           producer != rewrite::kInvalid)) {
        result.failure_code = 0xA7u;
        return result;
      }
      continue;
    }
    if (origin != rewrite::mixed_provenance::Origin::external ||
        producer != rewrite::kInvalid) {
      result.failure_code = 0xA8u;
      return result;
    }
    result.found = 1u;
    result.start = start;
    result.extent = end - start + 1u;
    result.generated_index = generated_index;
    result.generated_producer = generated_producer;
    result.distributed_generation = distributed_generation;
    result.value[0] = first_value;
    result.value[1] = second_value;
    result.value[2] = generated_value;
    result.value[3] = value;
    return result;
  }
  result.failure_code = 0xAEu;
  return result;
}

BCC32_PARTICIPATION_REAFFERENT_HD inline std::uint32_t
counter_exact_component(rewrite::ResidentRewriteState* state,
                        const participation::detail::ObservedTriplet& predicted) {
  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  // Worst-case bounded by the live population, not the fixed page-0
  // kRecordCapacity constant -- same reasoning and idiom as
  // resolve_relation_step in bcc32_resident_causal_constraint_participation.cuh.
  auto* sources =
      static_cast<std::uint32_t*>(malloc(sizeof(std::uint32_t) * capacity));
  if (sources == nullptr) return 0u;
  std::uint32_t source_count = 0u;
  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!participation::is_participation(record) || record.lane[5] == 0u ||
        record.lane[1] != participation::kConsequentFragment ||
        record.lane[2] != predicted.event[1].value ||
        record.lane[3] != predicted.event[2].value)
      continue;
    bool complete = false;
    for (std::uint32_t peer_slot = 0u;
         peer_slot < capacity; ++peer_slot) {
      const rewrite::Record& peer = state->records[peer_slot];
      complete |= participation::is_participation(peer) &&
          peer.lane[4] == record.lane[4] && peer.lane[5] != 0u &&
          peer.lane[2] == predicted.event[1].value &&
          peer.lane[1] == participation::kAntecedentFragment &&
          peer.lane[3] == predicted.event[0].value;
    }
    if (!complete)
      continue;
    bool seen = false;
    for (std::uint32_t prior = 0u; prior < source_count; ++prior)
      seen |= sources[prior] == record.lane[4];
    if (!seen)
      sources[source_count++] = record.lane[4];
  }
  std::uint32_t countered = 0u;
  for (std::uint32_t source = 0u; source < source_count; ++source)
    countered += participation::counter_source(state, sources[source]);
  free(sources);
  return countered;
}

// Re-derive the distributed closure that produced the staged event from the
// external prefix alone. This is deliberately a read-only query copy: it
// cannot publish, strengthen support, or turn the receipt into a new semantic
// cell. The generated suffix is excluded because the returned consequence must
// qualify the exact closure that existed before the action was emitted.
BCC32_PARTICIPATION_REAFFERENT_HD inline bool
distributed_generation_closure_authoritative(
    const rewrite::ResidentRewriteState* state,
    const rewrite::Record& trajectory,
    std::uint32_t* failure_code = nullptr) {
  if (state == nullptr || state->fault != 0u) {
    if (failure_code != nullptr) *failure_code = 0xd120u;
    return false;
  }
  const std::uint32_t generated_index =
      distributed_closure_anchor_index(state, trajectory);
  if (generated_index < 2u || generated_index >= trajectory.lane[2]) {
    if (failure_code != nullptr) *failure_code = 0xd121u;
    return false;
  }
  if ((trajectory.lane[7] & rewrite::kTrajectoryHasGenerated) == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd122u;
    return false;
  }
  if (trajectory.lane[3] != 0u) {
    if (failure_code != nullptr) *failure_code = 0xd123u;
    return false;
  }
  // An action return arrives on the active generated trajectory.  The yielded
  // marker belongs to the pause/continuation membrane and is not required for
  // a returned consequence to qualify the live distributed closure.
  if (generated_index == rewrite::kInvalid) {
    if (failure_code != nullptr) *failure_code = 0xd125u;
    return false;
  }
  if (state->causal_relation_generated_events == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd126u;
    return false;
  }
  if (state->causal_relation_component_digest == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd127u;
    return false;
  }
  if (state->causal_relation_component_revision_digest == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd128u;
    return false;
  }
  if (state->causal_relation_external_provenance_digest == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd129u;
    return false;
  }
  if (state->causal_relation_external_leaves == 0u) {
    if (failure_code != nullptr) *failure_code = 0xd12au;
    return false;
  }
  if (state->causal_relation_trajectory_revision == 0u ||
      trajectory.revision < state->causal_relation_trajectory_revision) {
    if (failure_code != nullptr) *failure_code = 0xd12bu;
    return false;
  }
  rewrite::Record query = trajectory;
  // The query digest is intentionally anchored to the exact trajectory
  // revision that produced the public event. Appending the returned external
  // consequence advances the live trajectory revision, but must not make a
  // valid chronology appear to be a different closure.
  query.revision = state->causal_relation_trajectory_revision;
  query.lane[2] = generated_index;
  query.lane[3] = 0u;
  query.lane[4] = 1u;
  query.lane[5] = rewrite::kInvalid;
  query.lane[7] = rewrite::kTrajectoryWasYielded;
  const rewrite::CausalRelationCandidate candidate =
      rewrite::collect_causal_relation_candidate(state, query);
  if (!candidate.ready || candidate.ambiguous) {
    if (failure_code != nullptr)
      *failure_code = candidate.ambiguous ? 0xd102u : 0xd101u;
    return false;
  }
  std::uint32_t generated_word = 0u;
  if (!rewrite::trajectory_word_at(state, trajectory.lane[1],
                                  generated_index, &generated_word)) {
    if (failure_code != nullptr) *failure_code = 0xd103u;
    return false;
  }
  if (candidate.word != generated_word) {
    if (failure_code != nullptr) *failure_code = 0xd104u;
    return false;
  }
  if (candidate.extent != generated_index) {
    if (failure_code != nullptr) *failure_code = 0xd105u;
    return false;
  }
  if (candidate.probe_steps != state->causal_relation_probe_steps) {
    if (failure_code != nullptr) *failure_code = 0xd106u;
    return false;
  }
  if (candidate.participating_records !=
      state->causal_relation_participating_records) {
    if (failure_code != nullptr) *failure_code = 0xd107u;
    return false;
  }
  if (candidate.independent_sources !=
      state->causal_relation_independent_sources) {
    if (failure_code != nullptr) *failure_code = 0xd108u;
    return false;
  }
  if (candidate.source_contributions !=
      state->causal_relation_source_contributions) {
    if (failure_code != nullptr) *failure_code = 0xd109u;
    return false;
  }
  if (candidate.maximum_source_contribution !=
      state->causal_relation_max_source_contribution) {
    if (failure_code != nullptr) *failure_code = 0xd10au;
    return false;
  }
  if (candidate.contribution_concentration_q16 !=
      state->causal_relation_contribution_concentration_q16) {
    if (failure_code != nullptr) *failure_code = 0xd10bu;
    return false;
  }
  if (candidate.singleton_supported_steps !=
      state->causal_relation_singleton_supported_steps) {
    if (failure_code != nullptr) *failure_code = 0xd10cu;
    return false;
  }
  if (candidate.minimum_probe_support !=
      state->causal_relation_minimum_probe_support) {
    if (failure_code != nullptr) *failure_code = 0xd10du;
    return false;
  }
  if (candidate.component_digest != state->causal_relation_component_digest) {
    if (failure_code != nullptr) *failure_code = 0xd10eu;
    return false;
  }
  if (candidate.component_revision_digest !=
      state->causal_relation_component_revision_digest) {
    if (failure_code != nullptr) *failure_code = 0xd10fu;
    return false;
  }
  if (candidate.external_provenance_digest !=
      state->causal_relation_external_provenance_digest) {
    if (failure_code != nullptr) *failure_code = 0xd110u;
    return false;
  }
  if (candidate.external_leaves != state->causal_relation_external_leaves) {
    if (failure_code != nullptr) *failure_code = 0xd111u;
    return false;
  }
  return true;
}
}  // namespace detail

// Invoke only on the already accepted private action-return staging world,
// immediately before its ordinary physical END. The caller supplies no ticket,
// relation, expected value, producer, source owner, or Record locus. All are
// recovered from one canonical external,external,generated/distributed+,external
// history.
BCC32_PARTICIPATION_REAFFERENT_HD inline Receipt
assimilate_accepted_staged_trajectory(rewrite::ResidentRewriteState* state) {
  Receipt receipt{};
  receipt.resident_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || state->fault != 0u)
    return receipt;
  const std::uint32_t trajectory_slot = rewrite::find_current_trajectory(state);
  if (trajectory_slot == rewrite::kInvalid)
    return receipt;
  receipt.staged_source_found = 1u;
  const rewrite::Record& trajectory = state->records[trajectory_slot];
  const detail::MixedProvenanceWindow window =
      detail::find_mixed_provenance_window(state, trajectory);
  if (window.omission != 0u) {
    receipt.provenance_exact = 1u;
    receipt.matched_window_start = window.start;
    receipt.matched_window_extent = window.extent;
    receipt.predicted_event = window.value[2];
    receipt.omission = 1u;
    receipt.predecessor_context = window.value[0];
    receipt.predecessor_antecedent = window.value[1];
    receipt.unresolved_residue = predictive_shadow::omission_residue_word(
        window.value[2], window.generated_index);
    receipt.rejected = 1u;
    receipt.rederived_event = 0xAEu;
    return receipt;
  }
  if (window.ambiguous != 0u || window.found == 0u) {
    receipt.rejected = 1u;
    // Preserve the legacy ambiguity code, but expose the observer-only
    // anchored-window failure branch when no window was found. This field is
    // diagnostic; admission authority remains the distributed receipt.
    receipt.rederived_event =
        window.ambiguous != 0u
            ? 0xAFu
            : (window.failure_code != 0u ? window.failure_code : 0xAEu);
    return receipt;
  }
  receipt.provenance_exact = 1u;
  receipt.matched_window_start = window.start;
  receipt.matched_window_extent = window.extent;
  receipt.predicted_event = window.value[2];
  receipt.returned_event = window.value[3];
  receipt.predecessor_context = window.value[0];
  receipt.predecessor_antecedent = window.value[1];

  bool ordinary_provider_exact = false;
  bool motor_babble_provider = false;
  if (window.distributed_generation != 0u) {
    receipt.generation_closure_rederived =
        detail::distributed_generation_closure_authoritative(
            state, trajectory, &receipt.rederived_event)
            ? 1u
            : 0u;
  } else {
    // A surviving span cursor is the exact resident provider identity. Its
    // owner, revision, and structural digest are revalidated together; the
    // historical provenance locus alone is never enough to authorize revision.
    bool provider_exact = false;
    if (detail::motor_babble_provider_exact(state, window)) {
      receipt.provider_locus = window.generated_producer;
      provider_exact = true;
      motor_babble_provider = true;
    }
    const std::uint32_t cursor_slot =
        rewrite::find_span_execution_cursor(state, trajectory.lane[1]);
    if (!provider_exact && cursor_slot != rewrite::kInvalid) {
      receipt.cursor_present = 1u;
      const rewrite::Record& cursor = state->records[cursor_slot];
      const std::uint32_t provider = rewrite::resolve_span_cursor_provider(
          state, cursor.lane[2], cursor.lane[7], cursor.reserved[0]);
      if (provider == window.generated_producer) {
        receipt.cursor_provider_exact = 1u;
        receipt.provider_locus = provider;
        provider_exact = true;
      }
    }
    if (!provider_exact &&
        !rewrite::span_execution_cursor_collision(
            state, trajectory.lane[1]) &&
        detail::fixed_program_provider_exact(
            state, trajectory, window)) {
      receipt.provider_locus = window.generated_producer;
      provider_exact = true;
    }
    if (provider_exact && !motor_babble_provider &&
        receipt.cursor_provider_exact == 0u)
      receipt.cursor_provider_exact = 1u;
    ordinary_provider_exact = provider_exact && !motor_babble_provider;
  }
  // Provider/cursor evidence is retained for diagnostics only. Production
  // admission is authorized exclusively by the distributed closure rederived
  // from live participation and external provenance; no provider locus can
  // authorize a returned consequence.
  // An exact ordinary provider proves which resident event was emitted; it
  // does not become future authority for the returned consequence. The
  // consequence is admitted below only as independently sourced participation
  // matter, while later execution must use its distributed component.
  if (window.distributed_generation == 0u &&
      (ordinary_provider_exact || motor_babble_provider))
    receipt.generation_closure_rederived = 1u;
  if (receipt.generation_closure_rederived == 0u) {
    receipt.rejected = 1u;
    if (receipt.rederived_event == 0u)
      receipt.rederived_event = window.distributed_generation != 0u ? 0xD1u
                                                                      : 0xC1u;
    return receipt;
  }
  if (window.value[2] == window.value[3]) {
    receipt.matched_return = 1u;
    receipt.rejected = 1u;
    return receipt;
  }

  participation::detail::ObservedTriplet predicted{};
  if (motor_babble_provider) {
    // The physical action itself is the learned body relation. The opaque
    // body mapping is outside cognition; only the actual returned consequence
    // enters as the observed after-state.
    predicted.event[0] = {window.value[2], 1u, rewrite::kEventFrameNone};
    predicted.event[1] = {window.value[2], 1u, rewrite::kEventFrameNone};
    predicted.event[2] = {window.value[2], 1u, rewrite::kEventFrameNone};
  } else {
    predicted.event[0] = {window.value[0], 1u, rewrite::kEventFrameNone};
    predicted.event[1] = {window.value[1], 1u, rewrite::kEventFrameNone};
    predicted.event[2] = {window.value[2], 1u, rewrite::kEventFrameNone};
  }
  participation::detail::ObservedTriplet observed{};
  observed.event[0] = predicted.event[0];
  observed.event[1] = predicted.event[1];
  observed.event[2] = {window.value[3], 1u, rewrite::kEventFrameNone};
  // Exact source is derived entirely from accepted canonical source owner and
  // the revalidated provider identity. It names no semantic relation.
  const std::uint32_t provider_revision =
      window.distributed_generation != 0u
          ? static_cast<std::uint32_t>(
                state->causal_relation_component_revision_digest != 0u
                    ? state->causal_relation_component_revision_digest
                    : state->revision)
      : state->records[window.generated_producer].revision;
  observed.source_revision = rewrite::rewrite_mix(
      trajectory.lane[1], window.distributed_generation != 0u
                              ? window.start
                              : window.generated_producer,
      provider_revision);
  receipt.source_revision = observed.source_revision;
  if (observed.source_revision == 0u ||
      observed.source_revision == rewrite::kInvalid ||
      rewrite::record_owner_exists(state, observed.source_revision)) {
    receipt.rejected = 1u;
    receipt.rederived_event = 0xD2u;
    return receipt;
  }
  // Reuse the proved raw predictive-shadow chemistry on this canonical
  // reafferent seam. The chronology index selects only a finite physical
  // rail; it is not a semantic category or global error signal.
  receipt.unresolved_residue = predictive_shadow::violation_residue_word(
      window.value[2], window.value[3], window.generated_index);
  receipt.local_reaction = predictive_shadow::local_contact_reaction_word(
      window.value[1], window.value[3], window.generated_index);
  receipt.predecessor_recontact =
      predictive_shadow::local_contact_reaction_word(
          window.value[0], window.value[1], window.generated_index);
  // Preflight source allocation before defeating old matter. The containing
  // action-return transaction remains atomic, but this ordering also keeps a
  // direct adapter caller from observing a half-applied revision.
  receipt.countered_records =
      motor_babble_provider
          ? participation::counter_conflicting_relation_consequences(
                state, window.value[2], window.value[2], window.value[3])
          : (window.distributed_generation != 0u &&
                     observed.event[2].value == predicted.event[2].value
                 ? 0u
                 : detail::counter_exact_component(
                       state, predicted));
  const participation::AssimilationReceipt admitted =
      participation::assimilate_validated_triplet(state, observed);
  receipt.admitted_records = admitted.admitted;
  receipt.rejected = admitted.rejected;
  if (receipt.rejected != 0u) receipt.rederived_event = 0xD3u;
  if (receipt.rejected == 0u) {
    // The returned external bytes are a new resident source for the exact
    // antecedent context. Retain only a compact, revalidated provenance
    // witness so later exact-context closure can survive trajectory withdrawal;
    // no returned word, answer, or semantic label is stored in the witness.
    std::uint64_t digest = source_witness::mix(
        0x13198a2e03707344ull, observed.source_revision);
    digest = source_witness::mix(digest, state->revision);
    for (std::uint32_t index = 0u; index < 3u; ++index) {
      digest = source_witness::mix(digest, index);
      digest = source_witness::mix(digest, observed.event[index].value);
      digest = source_witness::mix(digest, observed.event[index].reserved);
    }
    if (!source_witness::retain_external_source_digest(
            state, observed.source_revision, state->revision, digest, 3u,
            source_witness::kReafferentWitnessMarker)) {
      receipt.rejected = 1u;
      receipt.rederived_event = 0xd4u;
      return receipt;
    }
  }
  if (receipt.rejected == 0u) {
    // Re-read the component after the physical mutation. This observation is
    // never the admission authority: readiness remains the generic cut-closed,
    // unique live consequence law over resident participation fragments.
    // The returned consequence may be corroborated in a different antecedent
    // context. Re-derive candidate consequences from the relation's complete
    // live topology first; exact-context readout remains stricter and is still
    // used by ordinary generation. The return is only an observation used to
    // compare with that resident closure, never the closure's selector.
    const participation::StepResolution component =
        participation::resolve_cross_context_reafferent_component_from_relation(
            state, window.value[1]);
    const bool returned_component =
        component.ready != 0u && component.after == window.value[3];
    receipt.component_ready = returned_component ? 1u : 0u;
    receipt.component_ambiguous = component.ambiguous;
    receipt.component_records = component.participating_records;
    receipt.component_sources = component.independent_sources;
    receipt.rederived_event = component.after;
  }
  receipt.resident_revision = state->revision;
  return receipt;
}

}  // namespace substrate::bcc32::resident_causal_constraint_participation::reafferent

#undef BCC32_PARTICIPATION_REAFFERENT_HD
