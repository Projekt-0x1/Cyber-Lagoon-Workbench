#pragma once

#include <cstdint>

#include "causal_rewrite_universe.cuh"
#include "bcc32_resident_mixed_provenance_evidence.cuh"
#include "bcc32_resident_causal_relation_source_witness.cuh"

#if defined(__CUDACC__)
#define BCC32_PARTICIPATION_HD __host__ __device__
#else
#define BCC32_PARTICIPATION_HD
#endif

namespace substrate::bcc32::resident_causal_constraint_participation {

namespace rewrite = substrate::bcc32::causal_rewrite;
namespace source_witness =
    rewrite::resident_causal_relation_source_witness;

// One opaque physical form in the canonical Record ecology.  It is not a
// vocabulary, actor, task, answer, population, or executable Program identity.
// Each instance retains only one endpoint/relation fragment. No Record holds a
// complete local relation, and complete paths are never materialized: they
// exist only while a probe gathers overlapping independently sourced pairs.
inline constexpr std::uint32_t kFormConstraintParticipation =
    rewrite::kFormConstraintParticipation;
inline constexpr std::uint32_t kAntecedentFragment = 1u;
inline constexpr std::uint32_t kConsequentFragment = 2u;

inline constexpr std::uint32_t kAdapterValidateCanonicalSource = 1u << 0u;
inline constexpr std::uint32_t kAdapterAdmitAtPhysicalEnd = 1u << 1u;
inline constexpr std::uint32_t kAdapterRederiveBeforePublicWriter = 1u << 2u;
inline constexpr std::uint32_t kAdapterRefreshCanonicalReceipt = 1u << 3u;

// Integration manifest for the current ABI. The central form and owner seam
// are now registered; this packet still installs no second public writer. The
// coordinator-owned adapter must execute these phases in order and teach
// canonical owner allocation that lane 4 reserves the exact source owner for
// as long as participation survives.
struct IntegrationManifest {
  std::uint32_t physical_form = kFormConstraintParticipation;
  std::uint32_t required_record_lanes = rewrite::kLaneCount;
  std::uint32_t required_adapter_phases =
      kAdapterValidateCanonicalSource | kAdapterAdmitAtPhysicalEnd |
      kAdapterRederiveBeforePublicWriter | kAdapterRefreshCanonicalReceipt;
  std::uint32_t literal_observation_locus_only = 1u;
  std::uint32_t direct_projection_forbidden = 1u;
  std::uint32_t public_writer_owned_elsewhere = 1u;
  std::uint32_t central_form_registration_pending = 0u;
  std::uint32_t central_source_owner_reservation_pending = 0u;
};

inline constexpr IntegrationManifest current_main_integration_manifest() {
  return IntegrationManifest{};
}

BCC32_PARTICIPATION_HD constexpr bool current_main_form_id_disjoint() {
  return kFormConstraintParticipation != rewrite::kFormEmpty &&
      kFormConstraintParticipation != rewrite::kFormSequence &&
      kFormConstraintParticipation != rewrite::kFormDescription &&
      kFormConstraintParticipation != rewrite::kFormPartial &&
      kFormConstraintParticipation != rewrite::kFormMotor &&
      kFormConstraintParticipation != rewrite::kFormConstructor &&
      kFormConstraintParticipation != rewrite::kFormTrajectory &&
      kFormConstraintParticipation != rewrite::kFormTrajectoryTerm &&
      kFormConstraintParticipation != rewrite::kFormProgram &&
      kFormConstraintParticipation != rewrite::kFormProgramTerm &&
      kFormConstraintParticipation != rewrite::kFormSpanProgram &&
      kFormConstraintParticipation != rewrite::kFormSpanProgramTerm &&
      kFormConstraintParticipation != rewrite::kFormSpanExecutionCursor &&
      kFormConstraintParticipation != rewrite::kFormTransformationWitness &&
      kFormConstraintParticipation != rewrite::kFormProgramFactor &&
      kFormConstraintParticipation != rewrite::kFormProgramEvidence &&
      kFormConstraintParticipation != rewrite::kFormProgramAlternativeTerm &&
      kFormConstraintParticipation != rewrite::kFormProgramWitness &&
      kFormConstraintParticipation != rewrite::kFormConstructionEpisode &&
      kFormConstraintParticipation != rewrite::kFormConstructionEpisodeTerm &&
      kFormConstraintParticipation != rewrite::kFormCausalConstructor &&
      kFormConstraintParticipation != rewrite::kFormCausalConstructorTerm &&
      kFormConstraintParticipation != rewrite::kFormCausalConstructorWitness &&
      kFormConstraintParticipation != rewrite::kFormCausalConstructorDelta &&
      kFormConstraintParticipation != rewrite::kFormCausalProductWitness &&
      kFormConstraintParticipation != rewrite::kFormCausalCounterevidence &&
      kFormConstraintParticipation != rewrite::kFormRevisionTransferWitness &&
      kFormConstraintParticipation != rewrite::kFormRevisionTransferSourceUse &&
      kFormConstraintParticipation != rewrite::kFormTrajectoryProvenance;
}

static_assert(rewrite::kLaneCount == 8u,
              "participation requires the canonical eight-lane Record ABI");
static_assert(current_main_form_id_disjoint(),
              "participation form collides with current-main canonical matter");

// Internal material projection after canonical provenance validation. It is
// deliberately not exposed by a production kernel.
namespace detail {
struct ObservedTriplet {
  rewrite::RawRewriteEvent event[3]{};
  std::uint32_t source_revision = 0u;
};
}  // namespace detail

struct AssimilationReceipt {
  std::uint32_t admitted = 0u;
  std::uint32_t recurrent = 0u;
  std::uint32_t factor_intersections = 0u;
  std::uint32_t counterevidence = 0u;
  std::uint32_t turnover = 0u;
  std::uint32_t rejected = 0u;
  // Set only when allocate_participation_records() itself refused new
  // resident matter for a fully-validated triplet -- the sole discriminator
  // between genuine allocator pressure and every other (fail-closed,
  // non-pressure) rejection reason. A matter-paid caller may only authorize
  // turnover/retirement when this is set (0X1-207).
  std::uint32_t pressure_rejected = 0u;
  // Windows attempted by assimilate_completed_external_trajectory_window(s),
  // including a pressure-rejected one. Lets a bounded-work caller compute a
  // residual retry budget without granting a second full work_limit.
  std::uint32_t work_consumed = 0u;
  std::uint32_t retained_records = 0u;
  std::uint32_t free_records = 0u;
  std::uint64_t matter_q8 = 0u;
  std::uint64_t revision = 0u;
};

struct ProbeReceipt {
  std::uint32_t ready = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t steps_requested = 0u;
  std::uint32_t steps_completed = 0u;
  std::uint32_t proposed_event = 0u;
  std::uint32_t participating_records = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t positive_support = 0u;
  std::uint32_t counter_support = 0u;
  std::uint32_t weakest_step_support = 0u;
  std::uint32_t consequence_revision = 0u;
  std::uint64_t resident_revision = 0u;
};

struct StepResolution {
  std::uint32_t ready = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t after = 0u;
  std::uint32_t participating_records = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t positive_support = 0u;
  std::uint32_t counter_support = 0u;
  std::uint32_t consequence_revision = 0u;
};

// Read-only contribution for the existing span reader. It is deliberately
// not a Program, semantic route, cached path, or public writer receipt. The
// reader may compare this contribution with its other resident evidence; this
// component cannot append a word or publish an action by itself.
struct SpanReaderContribution {
  std::uint32_t ready = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t word = 0u;
  std::uint32_t extent = 0u;
  std::uint32_t participating_records = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t positive_support = 0u;
  std::uint32_t weakest_step_support = 0u;
  std::uint64_t resident_revision = 0u;
};

struct DamageReceipt {
  std::uint32_t requested = 0u;
  std::uint32_t displaced = 0u;
  std::uint32_t dispersed = 0u;
  std::uint32_t removed_matter_q8 = 0u;
};

struct InvariantReceipt {
  std::uint32_t ready = 0u;
  std::uint32_t checked_records = 0u;
  std::uint32_t valid_records = 0u;
  std::uint32_t complete_sources = 0u;
  std::uint32_t malformed_records = 0u;
  std::uint32_t orphan_fragments = 0u;
  std::uint32_t duplicate_fragments = 0u;
  std::uint32_t source_shape_conflicts = 0u;
  std::uint32_t source_polarity_conflicts = 0u;
  std::uint32_t matter_closed = 0u;
};

// One exact intersection between two independently accepted transaction
// fragments. This is an observer receipt over resident matter, not a new
// semantic Record, cached relation, source quorum, or public-output route.
// It retains only the shared fragment identity and exact contributing loci.
struct FactorIntersection {
  std::uint32_t fragment = 0u;
  std::uint32_t relation = 0u;
  std::uint32_t endpoint = 0u;
  std::uint32_t source_revision[2]{};
  std::uint32_t record_slot[2]{rewrite::kInvalid, rewrite::kInvalid};
  std::uint32_t record_revision[2]{};
  std::uint64_t resident_revision = 0u;
};

struct FactorFormationReceipt {
  std::uint32_t intersections = 0u;
  std::uint32_t emitted = 0u;
  std::uint32_t truncated = 0u;
  std::uint32_t rejected = 0u;
  std::uint64_t resident_revision = 0u;
};

BCC32_PARTICIPATION_HD inline bool valid_contact(
    const detail::ObservedTriplet& contact) {
  if (contact.source_revision == 0u)
    return false;
  for (std::uint32_t index = 0u; index < 3u; ++index)
    if (contact.event[index].valid == 0u ||
        contact.event[index].reserved != rewrite::kEventFrameNone ||
        contact.event[index].value == 0u ||
        contact.event[index].value == rewrite::kInvalid)
      return false;
  return true;
}

BCC32_PARTICIPATION_HD inline bool is_participation(
    const rewrite::Record& record) {
  return record.matter_q8 != 0u &&
      record.lane[0] == kFormConstraintParticipation &&
      (record.lane[1] == kAntecedentFragment ||
       record.lane[1] == kConsequentFragment) &&
      record.lane[2] != 0u && record.lane[2] != rewrite::kInvalid &&
      record.lane[3] != 0u && record.lane[3] != rewrite::kInvalid &&
      record.lane[4] != 0u &&
      ((record.lane[5] == 1u && record.lane[6] == 0u) ||
       (record.lane[5] == 0u && record.lane[6] == 1u)) &&
      record.lane[7] == 0u && record.reserved[0] == 0u &&
      record.reserved[1] == 0u;
}

// Structural/provenance audit only. Counts and digests are receipts; none is
// consulted by proposal selection. Every source must own exactly one positive
// or defeated antecedent fragment and one matching consequent fragment.
BCC32_PARTICIPATION_HD inline InvariantReceipt audit_invariants(
    const rewrite::ResidentRewriteState* state) {
  InvariantReceipt receipt{};
  if (state == nullptr)
    return receipt;
  receipt.matter_closed = rewrite::matter_account_is_closed(*state) ? 1u : 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormConstraintParticipation)
      continue;
    ++receipt.checked_records;
    if (!is_participation(record)) {
      ++receipt.malformed_records;
      continue;
    }
    ++receipt.valid_records;
    bool source_seen_earlier = false;
    for (std::uint32_t prior = 0u; prior < slot; ++prior)
      source_seen_earlier |= is_participation(state->records[prior]) &&
          state->records[prior].lane[4] == record.lane[4];
    if (source_seen_earlier)
      continue;
    std::uint32_t antecedents = 0u;
    std::uint32_t consequents = 0u;
    for (std::uint32_t peer_slot = 0u;
         peer_slot < rewrite::live_record_capacity(state); ++peer_slot) {
      const rewrite::Record& peer = state->records[peer_slot];
      if (!is_participation(peer) || peer.lane[4] != record.lane[4])
        continue;
      antecedents += peer.lane[1] == kAntecedentFragment ? 1u : 0u;
      consequents += peer.lane[1] == kConsequentFragment ? 1u : 0u;
      receipt.source_shape_conflicts += peer.lane[2] != record.lane[2] ? 1u : 0u;
      receipt.source_polarity_conflicts +=
          (peer.lane[5] != record.lane[5] ||
           peer.lane[6] != record.lane[6]) ? 1u : 0u;
    }
    if (antecedents == 1u && consequents == 1u) {
      ++receipt.complete_sources;
    } else {
      receipt.orphan_fragments +=
          antecedents == 0u || consequents == 0u ? 1u : 0u;
      receipt.duplicate_fragments +=
          antecedents > 1u ? antecedents - 1u : 0u;
      receipt.duplicate_fragments +=
          consequents > 1u ? consequents - 1u : 0u;
    }
  }
  receipt.ready = current_main_form_id_disjoint() &&
      receipt.matter_closed != 0u && receipt.malformed_records == 0u &&
      receipt.orphan_fragments == 0u && receipt.duplicate_fragments == 0u &&
      receipt.source_shape_conflicts == 0u &&
      receipt.source_polarity_conflicts == 0u ? 1u : 0u;
  return receipt;
}

BCC32_PARTICIPATION_HD inline bool same_fragment_identity(
    const rewrite::Record& record, const detail::ObservedTriplet& contact,
    std::uint32_t fragment) {
  const std::uint32_t endpoint = fragment == kAntecedentFragment
      ? contact.event[0].value : contact.event[2].value;
  return is_participation(record) &&
      record.lane[1] == fragment &&
      record.lane[2] == contact.event[1].value &&
      record.lane[3] == endpoint &&
      record.lane[4] == contact.source_revision;
}

BCC32_PARTICIPATION_HD inline bool same_contact_fragment(
    const rewrite::Record& record, const detail::ObservedTriplet& contact,
    std::uint32_t fragment) {
  return same_fragment_identity(record, contact, fragment) &&
      record.lane[5] == 1u;
}

// A fragment participates in the passive factor graph only while its exact
// source still owns one positive antecedent/consequent pair. This rederives
// the companion rather than trusting a copied receipt or a locally plausible
// Record, so defeated, orphaned, duplicated, and forged matter contributes no
// lineage.
BCC32_PARTICIPATION_HD inline bool is_accepted_transaction_fragment(
    const rewrite::ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot >= rewrite::live_record_capacity(state))
    return false;
  const rewrite::Record& fragment = state->records[slot];
  if (!is_participation(fragment) || fragment.lane[5] != 1u ||
      fragment.lane[6] != 0u)
    return false;
  std::uint32_t same_kind = 0u;
  std::uint32_t companion_kind = 0u;
  for (std::uint32_t peer_slot = 0u;
       peer_slot < rewrite::live_record_capacity(state); ++peer_slot) {
    const rewrite::Record& peer = state->records[peer_slot];
    if (!is_participation(peer) || peer.lane[4] != fragment.lane[4])
      continue;
    if (peer.lane[2] != fragment.lane[2] || peer.lane[5] != 1u ||
        peer.lane[6] != 0u)
      return false;
    if (peer.lane[1] == fragment.lane[1])
      ++same_kind;
    else
      ++companion_kind;
  }
  return same_kind == 1u && companion_kind == 1u;
}

BCC32_PARTICIPATION_HD inline bool same_factor_fragment(
    const rewrite::Record& left, const rewrite::Record& right) {
  return left.lane[1] == right.lane[1] &&
      left.lane[2] == right.lane[2] && left.lane[3] == right.lane[3];
}

BCC32_PARTICIPATION_HD inline FactorIntersection factor_intersection(
    const rewrite::ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot) {
  FactorIntersection result{};
  if (!is_accepted_transaction_fragment(state, left_slot) ||
      !is_accepted_transaction_fragment(state, right_slot))
    return result;
  const rewrite::Record& left = state->records[left_slot];
  const rewrite::Record& right = state->records[right_slot];
  if (left.lane[4] == right.lane[4] || !same_factor_fragment(left, right))
    return result;
  const bool left_first = left.lane[4] < right.lane[4];
  const std::uint32_t first_slot = left_first ? left_slot : right_slot;
  const std::uint32_t second_slot = left_first ? right_slot : left_slot;
  const rewrite::Record& first = state->records[first_slot];
  const rewrite::Record& second = state->records[second_slot];
  result.fragment = first.lane[1];
  result.relation = first.lane[2];
  result.endpoint = first.lane[3];
  result.source_revision[0] = first.lane[4];
  result.source_revision[1] = second.lane[4];
  result.record_slot[0] = first_slot;
  result.record_slot[1] = second_slot;
  result.record_revision[0] = first.revision;
  result.record_revision[1] = second.revision;
  result.resident_revision = state->revision;
  return result;
}

BCC32_PARTICIPATION_HD inline bool factor_intersection_is_current(
    const rewrite::ResidentRewriteState* state,
    const FactorIntersection& candidate) {
  if (state == nullptr || candidate.fragment == 0u ||
      candidate.record_slot[0] >= rewrite::live_record_capacity(state) ||
      candidate.record_slot[1] >= rewrite::live_record_capacity(state) ||
      candidate.resident_revision != state->revision)
    return false;
  const FactorIntersection current = factor_intersection(
      state, candidate.record_slot[0], candidate.record_slot[1]);
  return current.fragment == candidate.fragment &&
      current.relation == candidate.relation &&
      current.endpoint == candidate.endpoint &&
      current.source_revision[0] == candidate.source_revision[0] &&
      current.source_revision[1] == candidate.source_revision[1] &&
      current.record_slot[0] == candidate.record_slot[0] &&
      current.record_slot[1] == candidate.record_slot[1] &&
      current.record_revision[0] == candidate.record_revision[0] &&
      current.record_revision[1] == candidate.record_revision[1] &&
      current.resident_revision == candidate.resident_revision;
}

BCC32_PARTICIPATION_HD inline bool factor_intersection_logically_less(
    const FactorIntersection& left, const FactorIntersection& right) {
  if (left.fragment != right.fragment)
    return left.fragment < right.fragment;
  if (left.relation != right.relation)
    return left.relation < right.relation;
  if (left.endpoint != right.endpoint)
    return left.endpoint < right.endpoint;
  if (left.source_revision[0] != right.source_revision[0])
    return left.source_revision[0] < right.source_revision[0];
  return left.source_revision[1] < right.source_revision[1];
}

BCC32_PARTICIPATION_HD inline void insert_factor_intersection_prefix(
    FactorIntersection* output, std::uint32_t output_capacity,
    std::uint32_t* emitted, const FactorIntersection& candidate) {
  if (output == nullptr || emitted == nullptr || output_capacity == 0u)
    return;
  if (*emitted < output_capacity) {
    output[(*emitted)++] = candidate;
  } else if (factor_intersection_logically_less(
                 candidate, output[output_capacity - 1u])) {
    output[output_capacity - 1u] = candidate;
  } else {
    return;
  }
  std::uint32_t index = *emitted < output_capacity
      ? *emitted - 1u : output_capacity - 1u;
  while (index != 0u && factor_intersection_logically_less(
                             output[index], output[index - 1u])) {
    const FactorIntersection prior = output[index - 1u];
    output[index - 1u] = output[index];
    output[index] = prior;
    --index;
  }
}

BCC32_PARTICIPATION_HD inline FactorFormationReceipt form_factor_intersections(
    const rewrite::ResidentRewriteState* state, FactorIntersection* output,
    std::uint32_t output_capacity) {
  FactorFormationReceipt receipt{};
  if (state == nullptr || state->fault != 0u ||
      (output_capacity != 0u && output == nullptr)) {
    receipt.rejected = 1u;
    return receipt;
  }
  receipt.resident_revision = state->revision;
  for (std::uint32_t left = 0u; left < rewrite::live_record_capacity(state); ++left) {
    if (!is_accepted_transaction_fragment(state, left))
      continue;
    for (std::uint32_t right = left + 1u;
         right < rewrite::live_record_capacity(state); ++right) {
      const FactorIntersection intersection =
          factor_intersection(state, left, right);
      if (intersection.fragment == 0u)
        continue;
      ++receipt.intersections;
      insert_factor_intersection_prefix(
          output, output_capacity, &receipt.emitted, intersection);
    }
  }
  receipt.truncated = receipt.emitted != receipt.intersections ? 1u : 0u;
  return receipt;
}

// Count-only path for the physical-END adapter. It exposes the passive
// formation measurement without constructing a potentially large receipt on
// the production close stack. The count is never consulted by authority.
BCC32_PARTICIPATION_HD inline std::uint32_t count_factor_intersections(
    const rewrite::ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u)
    return 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t left = 0u; left < rewrite::live_record_capacity(state); ++left) {
    if (!is_accepted_transaction_fragment(state, left))
      continue;
    const rewrite::Record& left_record = state->records[left];
    for (std::uint32_t right = left + 1u;
         right < rewrite::live_record_capacity(state); ++right) {
      if (!is_accepted_transaction_fragment(state, right))
        continue;
      const rewrite::Record& right_record = state->records[right];
      count += left_record.lane[4] != right_record.lane[4] &&
          same_factor_fragment(left_record, right_record) ? 1u : 0u;
    }
  }
  return count;
}

BCC32_PARTICIPATION_HD inline bool record_logically_less(
    const rewrite::Record& left, const rewrite::Record& right) {
  for (std::uint32_t lane = 1u; lane < rewrite::kLaneCount; ++lane) {
    if (left.lane[lane] != right.lane[lane])
      return left.lane[lane] < right.lane[lane];
  }
  return left.revision < right.revision;
}

// Select all matter needed by one contact before mutating any Record. Empty
// matter is preferred. Under complete pressure, only defeated participation
// fragments may turn over; unrelated canonical Records are never reclaimed.
BCC32_PARTICIPATION_HD inline bool allocate_participation_records(
    rewrite::ResidentRewriteState* state, std::uint32_t needed,
    std::uint32_t* selected_slots, std::uint32_t* turned_over) {
  if (state == nullptr || selected_slots == nullptr || needed == 0u ||
      needed > 2u)
    return false;
  std::uint32_t selected_count = 0u;
  while (selected_count < needed) {
    const std::uint32_t start = selected_count == 0u
        ? state->allocation_cursor
        : (selected_slots[0] + rewrite::live_record_capacity(state) / 2u) %
            rewrite::live_record_capacity(state);
    bool found = false;
    for (std::uint32_t offset = 0u; offset < rewrite::live_record_capacity(state);
         ++offset) {
      const std::uint32_t slot =
          (start + offset) % rewrite::live_record_capacity(state);
      bool already_selected = false;
      for (std::uint32_t prior = 0u; prior < selected_count; ++prior)
        already_selected |= selected_slots[prior] == slot;
      const rewrite::Record& record = state->records[slot];
      if (already_selected || record.matter_q8 == 0u ||
          record.lane[0] != rewrite::kFormEmpty)
        continue;
      selected_slots[selected_count++] = slot;
      found = true;
      break;
    }
    if (!found)
      break;
  }
  if (selected_count < needed) {
    // Never combine spare matter with half of another defeated source. Under
    // full pressure, replace exactly one structurally complete defeated pair.
    if (selected_count != 0u || needed != 2u)
      return false;
    std::uint32_t selected_antecedent = rewrite::kInvalid;
    std::uint32_t selected_consequent = rewrite::kInvalid;
    for (std::uint32_t antecedent_slot = 0u;
         antecedent_slot < rewrite::live_record_capacity(state); ++antecedent_slot) {
      const rewrite::Record& antecedent = state->records[antecedent_slot];
      if (!is_participation(antecedent) ||
          antecedent.lane[1] != kAntecedentFragment ||
          antecedent.lane[5] != 0u || antecedent.lane[6] != 1u)
        continue;
      std::uint32_t consequent_slot = rewrite::kInvalid;
      std::uint32_t consequent_count = 0u;
      for (std::uint32_t peer_slot = 0u;
           peer_slot < rewrite::live_record_capacity(state); ++peer_slot) {
        const rewrite::Record& peer = state->records[peer_slot];
        if (!is_participation(peer) ||
            peer.lane[1] != kConsequentFragment ||
            peer.lane[2] != antecedent.lane[2] ||
            peer.lane[4] != antecedent.lane[4] ||
            peer.lane[5] != 0u || peer.lane[6] != 1u)
          continue;
        consequent_slot = peer_slot;
        ++consequent_count;
      }
      if (consequent_count != 1u)
        continue;
      const bool earlier_pair = selected_antecedent == rewrite::kInvalid ||
          record_logically_less(
              antecedent, state->records[selected_antecedent]) ||
          (!record_logically_less(
               state->records[selected_antecedent], antecedent) &&
           record_logically_less(
               state->records[consequent_slot],
               state->records[selected_consequent]));
      if (earlier_pair) {
        selected_antecedent = antecedent_slot;
        selected_consequent = consequent_slot;
      }
    }
    if (selected_antecedent == rewrite::kInvalid ||
        selected_consequent == rewrite::kInvalid)
      return false;
    selected_slots[0] = selected_antecedent;
    selected_slots[1] = selected_consequent;
    selected_count = 2u;
  }
  for (std::uint32_t index = 0u; index < needed; ++index) {
    rewrite::Record& record = state->records[selected_slots[index]];
    if (record.lane[0] != rewrite::kFormEmpty) {
      rewrite::clear_record(&record);
      if (turned_over != nullptr)
        ++*turned_over;
    }
  }
  state->allocation_cursor =
      (selected_slots[needed - 1u] + 1u) % rewrite::live_record_capacity(state);
  return true;
}

BCC32_PARTICIPATION_HD inline AssimilationReceipt
assimilate_validated_triplet(
    rewrite::ResidentRewriteState* state,
    const detail::ObservedTriplet& contact,
    bool measure_factor_intersections = true,
    bool measure_resident_summary = true) {
  AssimilationReceipt receipt{};
  if (state == nullptr || state->fault != 0u || !valid_contact(contact)) {
    receipt.rejected = 1u;
    return receipt;
  }
  std::uint32_t missing[2]{};
  std::uint32_t missing_count = 0u;
  bool found[2]{};
  // Both fragment identities live in the same resident aperture.  Fuse their
  // existential/conflict scan so one window pays for one physical pass rather
  // than one pass per fragment; this does not introduce an index or semantic
  // selector and preserves the same fail-closed identity predicates.
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(state); ++slot) {
    for (std::uint32_t fragment = kAntecedentFragment;
         fragment <= kConsequentFragment; ++fragment) {
      if (same_fragment_identity(state->records[slot], contact, fragment) &&
          !same_contact_fragment(state->records[slot], contact, fragment)) {
        receipt.rejected = 1u;
        return receipt;
      }
      found[fragment - kAntecedentFragment] |=
          same_contact_fragment(state->records[slot], contact, fragment);
    }
  }
  for (std::uint32_t fragment = kAntecedentFragment;
       fragment <= kConsequentFragment; ++fragment) {
    if (!found[fragment - kAntecedentFragment])
      missing[missing_count++] = fragment;
  }
  receipt.recurrent = missing_count < 2u ? 1u : 0u;
  if (missing_count != 0u) {
    std::uint32_t selected[2]{};
    if (!allocate_participation_records(
            state, missing_count, selected, &receipt.turnover)) {
      receipt.pressure_rejected = 1u;
      receipt.rejected = 1u;
    } else for (std::uint32_t index = 0u; index < missing_count; ++index) {
      rewrite::Record& record = state->records[selected[index]];
      const std::uint32_t matter = record.matter_q8;
      const std::uint32_t revision = record.revision + 1u;
      record = rewrite::Record{};
      record.lane[0] = kFormConstraintParticipation;
      record.lane[1] = missing[index];
      record.lane[2] = contact.event[1].value;
      record.lane[3] = missing[index] == kAntecedentFragment
          ? contact.event[0].value : contact.event[2].value;
      record.lane[4] = contact.source_revision;
      record.lane[5] = 1u;
      record.lane[6] = 0u;
      record.revision = revision;
      record.matter_q8 = matter;
      ++state->revision;
      ++receipt.admitted;
      receipt.revision = state->revision;
    }
  }
  if (measure_resident_summary) {
    for (std::uint32_t slot = 0u;
         slot < rewrite::live_record_capacity(state); ++slot) {
      const rewrite::Record& record = state->records[slot];
      receipt.matter_q8 += record.matter_q8;
      if (is_participation(record))
        ++receipt.retained_records;
      else if (record.matter_q8 != 0u &&
               record.lane[0] == rewrite::kFormEmpty)
        ++receipt.free_records;
    }
  }
  // Factor formation is an observer receipt, not an admission predicate. The
  // resumable physical-END window path can process many overlapping windows;
  // recomputing the complete resident factor population for every one turns
  // bounded admission into a capacity-cubed measurement loop. Keep the
  // canonical/default path fully observable, while allowing that production
  // adapter to defer this non-authoritative count.
  if (measure_factor_intersections)
    receipt.factor_intersections = count_factor_intersections(state);
  // The next free-matter search is an opaque physical placement decision. It
  // derives from the complete lived source, never from an answer, route, or
  // relation category. This prevents consecutive episodes of one recurring
  // local constraint from becoming one contiguous physical failure domain.
  // Exact lanes remain the sole authority; this value is only an allocator
  // cursor and allocation permutation remains an explicit control.
  state->allocation_cursor = rewrite::rewrite_mix(
      rewrite::rewrite_mix(contact.source_revision, contact.event[0].value,
                           contact.event[1].value),
      contact.event[2].value, kFormConstraintParticipation) %
      rewrite::live_record_capacity(state);
  return receipt;
}

// Production admission derives content and provenance from one canonical
// literal observation. The caller supplies only its resident locus; it cannot
// choose endpoints, relation, polarity, expected output, or a semantic route.
BCC32_PARTICIPATION_HD inline AssimilationReceipt
assimilate_literal_observation(rewrite::ResidentRewriteState* state,
                               std::uint32_t source_slot) {
  AssimilationReceipt rejected{};
  rejected.rejected = 1u;
  if (state == nullptr || source_slot >= rewrite::live_record_capacity(state) ||
      !rewrite::literal_observation_source(state, source_slot))
    return rejected;
  const rewrite::Record& source = state->records[source_slot];
  if (source.lane[2] != 3u || source.lane[1] == 0u ||
      source.lane[1] == rewrite::kInvalid)
    return rejected;
  if (source.lane[0] == rewrite::kFormTrajectory) {
    if (source.lane[3] == 0u ||
        !rewrite::mixed_provenance::tagged_history(state, source))
      return rejected;
    for (std::uint32_t index = 0u; index < 3u; ++index) {
      rewrite::mixed_provenance::Origin origin{};
      std::uint32_t producer = rewrite::kInvalid;
      if (!rewrite::mixed_provenance::origin_at(
              state, source, index, &origin, &producer) ||
          origin != rewrite::mixed_provenance::Origin::external ||
          producer != rewrite::kInvalid)
        return rejected;
    }
  }
  detail::ObservedTriplet contact{};
  for (std::uint32_t index = 0u; index < 3u; ++index) {
    std::uint32_t value = 0u;
    if (!rewrite::literal_observation_word_at_prevalidated(
            state, source_slot, index, &value))
      return rejected;
    contact.event[index] = rewrite::RawRewriteEvent{
        value, 1u, rewrite::kEventFrameNone};
  }
  contact.source_revision = source.lane[1];
  return assimilate_validated_triplet(state, contact);
}

// Resolve a trajectory term after the complete source-validation cursor has
// already established external provenance for every event. The overlapping
// window reader visits the sequence [n,n+1,n+2], then [n+1,n+2,n+3]; retain the
// two most recent physical term slots so only the newly entering term scans
// the resident aperture. Cache hits still recheck the ordinary Record shape,
// owner, and ordinal, and this helper is never used by the default
// revalidating path.
BCC32_PARTICIPATION_HD inline bool cached_trajectory_term_matches(
    const rewrite::ResidentRewriteState* state, std::uint32_t page,
    std::uint32_t owner, std::uint32_t ordinal, std::uint32_t cached_page,
    std::uint32_t cached_ordinal, std::uint32_t cached_slot) {
  if (state == nullptr || cached_page != page ||
      cached_ordinal != ordinal || cached_slot == rewrite::kInvalid)
    return false;
  const rewrite::Record& term = state->records[cached_slot];
  return term.matter_q8 != 0u &&
      term.lane[0] == rewrite::kFormTrajectoryTerm &&
      term.lane[1] == owner && term.lane[2] == ordinal;
}

BCC32_PARTICIPATION_HD inline bool prevalidated_trajectory_word_at(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr || owner == 0u ||
      owner == rewrite::kInvalid)
    return false;
  const std::uint32_t page = index / rewrite::kTrajectoryPageEvents;
  if (page >= rewrite::live_record_capacity(state)) return false;
  const std::uint32_t local = index % rewrite::kTrajectoryPageEvents;
  std::uint32_t term_owner = owner;
  if (page != 0u) {
    std::uint32_t page_slot = state->external_relation_stage_page_slot;
    if (state->external_relation_stage_page != page ||
        page_slot == rewrite::kInvalid) {
      page_slot = rewrite::find_owned_block(
          state, rewrite::kFormTrajectoryPage, owner, page);
      state->external_relation_stage_page = page;
      state->external_relation_stage_page_slot = page_slot;
    }
    if (page_slot == rewrite::kInvalid) return false;
    const rewrite::Record& continuation = state->records[page_slot];
    if (continuation.lane[3] != page * rewrite::kTrajectoryPageEvents ||
        continuation.lane[4] <= local ||
        continuation.lane[4] > rewrite::kTrajectoryPageEvents ||
        continuation.lane[6] == 0u || continuation.lane[6] == rewrite::kInvalid)
      return false;
    term_owner = continuation.lane[6];
  }
  const std::uint32_t ordinal = local / 2u;
  std::uint32_t term_slot = rewrite::kInvalid;
  if (cached_trajectory_term_matches(
          state, page, term_owner, ordinal,
          state->external_relation_stage_term_page,
          state->external_relation_stage_term_ordinal,
          state->external_relation_stage_term_slot)) {
    term_slot = state->external_relation_stage_term_slot;
  } else if (cached_trajectory_term_matches(
                 state, page, term_owner, ordinal,
                 state->external_relation_stage_term_alt_page,
                 state->external_relation_stage_term_alt_ordinal,
                 state->external_relation_stage_term_alt_slot)) {
    term_slot = state->external_relation_stage_term_alt_slot;
  } else {
    term_slot = rewrite::find_owned_block(
        state, rewrite::kFormTrajectoryTerm, term_owner, ordinal);
    state->external_relation_stage_term_alt_page =
        state->external_relation_stage_term_page;
    state->external_relation_stage_term_alt_ordinal =
        state->external_relation_stage_term_ordinal;
    state->external_relation_stage_term_alt_slot =
        state->external_relation_stage_term_slot;
    state->external_relation_stage_term_page = page;
    state->external_relation_stage_term_ordinal = ordinal;
    state->external_relation_stage_term_slot = term_slot;
  }
  if (term_slot == rewrite::kInvalid) return false;
  *word = state->records[term_slot].lane[4u + (local % 2u)];
  return true;
}

// Admit one overlapping local constraint from a complete external trajectory.
// The three-event window is the bootstrap interaction arity only; callers may
// advance `window_start` across an arbitrary page-native source. No complete
// source, phrase, answer, or semantic relation is copied into a Record.
BCC32_PARTICIPATION_HD inline AssimilationReceipt
assimilate_completed_external_trajectory_window(
    rewrite::ResidentRewriteState* state, std::uint32_t source_slot,
    std::uint32_t window_start, bool require_provenance = true) {
  AssimilationReceipt rejected{};
  rejected.rejected = 1u;
  if (state == nullptr || source_slot >= rewrite::live_record_capacity(state))
    return rejected;
  const rewrite::Record& source = state->records[source_slot];
  if (source.matter_q8 == 0u ||
      source.lane[0] != rewrite::kFormTrajectory ||
      source.lane[1] == 0u || source.lane[1] == rewrite::kInvalid ||
      source.lane[2] < 3u || window_start > source.lane[2] - 3u ||
      source.lane[3] != 0u || source.lane[4] != 0u ||
      source.lane[5] != rewrite::kInvalid || source.lane[7] != 0u)
    return rejected;
  detail::ObservedTriplet contact{};
  for (std::uint32_t local = 0u; local < 3u; ++local) {
    std::uint32_t value = 0u;
    if (require_provenance) {
      rewrite::mixed_provenance::Origin origin{};
      std::uint32_t producer = rewrite::kInvalid;
      const bool valid = rewrite::mixed_provenance::origin_at(
                              state, source, window_start + local, &origin,
                              &producer) &&
          origin == rewrite::mixed_provenance::Origin::external &&
          producer == rewrite::kInvalid &&
          rewrite::trajectory_word_at(
              state, source.lane[1], window_start + local, &value);
      if (!valid) return rejected;
    } else if (!prevalidated_trajectory_word_at(
                   state, source.lane[1], window_start + local, &value)) {
      return rejected;
    }
    contact.event[local] = rewrite::RawRewriteEvent{
        value, 1u, rewrite::kEventFrameNone};
  }
  // The owner is physical provenance identity, not a relation or source
  // selector supplied by the caller.
  contact.source_revision = source.lane[1];
  return require_provenance
      ? assimilate_validated_triplet(state, contact, false)
      : assimilate_validated_triplet(state, contact, false, false);
}

// Consume a bounded number of windows and resume from the returned cursor.
// `work_limit` bounds execution only; it is never a logical source-length or
// language-context limit. A later close epoch observes the exact same source
// owner and window order through the resident state.
BCC32_PARTICIPATION_HD inline AssimilationReceipt
assimilate_completed_external_trajectory_windows(
    rewrite::ResidentRewriteState* state, std::uint32_t source_slot,
    std::uint32_t* next_window, std::uint32_t work_limit,
    bool prevalidated = false) {
  AssimilationReceipt aggregate{};
  if (state == nullptr || next_window == nullptr || work_limit == 0u ||
      source_slot >= rewrite::live_record_capacity(state)) {
    aggregate.rejected = 1u;
    return aggregate;
  }
  const rewrite::Record& source = state->records[source_slot];
  if (source.matter_q8 == 0u || source.lane[2] < 3u) {
    aggregate.rejected = 1u;
    return aggregate;
  }
  std::uint32_t work = 0u;
  while (*next_window <= source.lane[2] - 3u && work < work_limit) {
    const AssimilationReceipt receipt =
        assimilate_completed_external_trajectory_window(
            state, source_slot, *next_window, !prevalidated);
    ++work;
    if (receipt.rejected != 0u) {
      aggregate.rejected = 1u;
      // Propagate whether this specific window's rejection was genuine
      // resident-matter allocator pressure (as opposed to malformed input or
      // provenance mismatch), and how many windows this call attempted --
      // the two signals a matter-paid caller needs to authorize turnover and
      // compute a residual retry budget without exceeding work_limit (0X1-207).
      aggregate.pressure_rejected = receipt.pressure_rejected;
      aggregate.work_consumed = work;
      return aggregate;
    }
    aggregate.admitted += receipt.admitted;
    aggregate.recurrent += receipt.recurrent;
    aggregate.factor_intersections += receipt.factor_intersections;
    aggregate.counterevidence += receipt.counterevidence;
    aggregate.turnover += receipt.turnover;
    ++*next_window;
  }
  aggregate.work_consumed = work;
  aggregate.revision = state->revision;
  return aggregate;
}

// Compatibility seam retained for the original three-word donor contract.
// Production physical END uses the resumable window adapter above.
BCC32_PARTICIPATION_HD inline AssimilationReceipt
assimilate_completed_external_trajectory_at_end(
    rewrite::ResidentRewriteState* state, std::uint32_t source_slot) {
  AssimilationReceipt rejected{};
  rejected.rejected = 1u;
  if (state == nullptr || source_slot >= rewrite::live_record_capacity(state))
    return rejected;
  const rewrite::Record& source = state->records[source_slot];
  if (source.matter_q8 == 0u ||
      source.lane[0] != rewrite::kFormTrajectory || source.lane[2] != 3u ||
      source.lane[3] != 0u || source.lane[4] != 0u ||
      source.lane[5] != rewrite::kInvalid || source.lane[7] != 0u ||
      source.lane[1] == 0u || source.lane[1] == rewrite::kInvalid ||
      !rewrite::mixed_provenance::tagged_history(state, source))
    return rejected;
  return assimilate_completed_external_trajectory_window(state, source_slot, 0u);
}

// Counterevidence is a provenance transition of one exact source, never a
// caller-supplied negative label on arbitrary content. It defeats but retains
// the fragments so ordinary matter pressure can later turn them over.
BCC32_PARTICIPATION_HD inline std::uint32_t counter_source(
    rewrite::ResidentRewriteState* state, std::uint32_t source_revision) {
  if (state == nullptr || source_revision == 0u)
    return 0u;
  std::uint32_t countered = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    rewrite::Record& record = state->records[slot];
    if (!is_participation(record) || record.lane[4] != source_revision ||
        record.lane[5] == 0u)
      continue;
    record.lane[5] = 0u;
    record.lane[6] = 1u;
    ++record.revision;
    ++state->revision;
    ++countered;
  }
  return countered;
}

BCC32_PARTICIPATION_HD inline bool paired_antecedent_exists(
    const rewrite::ResidentRewriteState* state,
    const rewrite::Record& consequent, std::uint32_t before,
    bool positive) {
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& antecedent = state->records[slot];
    if (!is_participation(antecedent) ||
        antecedent.lane[1] != kAntecedentFragment ||
        antecedent.lane[2] != consequent.lane[2] ||
        antecedent.lane[3] != before ||
        antecedent.lane[4] != consequent.lane[4])
      continue;
    if (positive ? antecedent.lane[5] != 0u
                 : antecedent.lane[6] != 0u)
      return true;
  }
  return false;
}

// Re-derive uniqueness from the preceding live lineage instead of allocating
// a capacity-sized candidate table in every relation step. The scan order is
// the resident slot order, so this preserves the first-seen alternative order
// while keeping the reader's authority entirely in live Records.
BCC32_PARTICIPATION_HD inline bool relation_alternative_seen_before(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t alternative,
    std::uint32_t current_slot) {
  for (std::uint32_t slot = 0u; slot < current_slot; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) ||
        record.lane[1] != kConsequentFragment ||
        record.lane[2] != relation || record.lane[3] != alternative ||
        (!paired_antecedent_exists(state, record, before, true) &&
         !paired_antecedent_exists(state, record, before, false)))
      continue;
    return true;
  }
  return false;
}

BCC32_PARTICIPATION_HD inline bool relation_positive_source_seen_before(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after, std::uint32_t source,
    std::uint32_t current_slot) {
  for (std::uint32_t slot = 0u; slot < current_slot; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) ||
        record.lane[1] != kConsequentFragment ||
        record.lane[2] != relation || record.lane[3] != after ||
        record.lane[4] != source || record.lane[5] == 0u ||
        !paired_antecedent_exists(state, record, before, true))
      continue;
    return true;
  }
  return false;
}

// A new accepted body consequence may contradict an older complete source
// for the same resident actuator relation. Counter only incompatible complete
// sources; the new observation still has to pass the ordinary distributed
// assimilation and cut-closure checks before it can influence later output.
BCC32_PARTICIPATION_HD inline std::uint32_t
counter_conflicting_relation_consequences(
    rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t observed_after) {
  if (state == nullptr || before == 0u || relation == 0u ||
      observed_after == 0u || before == rewrite::kInvalid ||
      relation == rewrite::kInvalid || observed_after == rewrite::kInvalid)
    return 0u;
  std::uint32_t countered = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& consequent = state->records[slot];
    if (!is_participation(consequent) ||
        consequent.lane[1] != kConsequentFragment ||
        consequent.lane[2] != relation || consequent.lane[5] == 0u ||
        consequent.lane[3] == observed_after ||
        !paired_antecedent_exists(state, consequent, before, true))
      continue;
    countered += counter_source(state, consequent.lane[4]);
  }
  return countered;
}

BCC32_PARTICIPATION_HD inline bool positive_pair_exists_excluding(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after,
    std::uint32_t excluded_slot) {
  for (std::uint32_t consequent_slot = 0u;
       consequent_slot < rewrite::live_record_capacity(state); ++consequent_slot) {
    const rewrite::Record& consequent = state->records[consequent_slot];
    if (consequent_slot == excluded_slot || !is_participation(consequent) ||
        consequent.lane[1] != kConsequentFragment ||
        consequent.lane[2] != relation || consequent.lane[3] != after ||
        consequent.lane[5] == 0u)
      continue;
    for (std::uint32_t antecedent_slot = 0u;
         antecedent_slot < rewrite::live_record_capacity(state); ++antecedent_slot) {
      const rewrite::Record& antecedent = state->records[antecedent_slot];
      if (antecedent_slot == excluded_slot ||
          !is_participation(antecedent) ||
          antecedent.lane[1] != kAntecedentFragment ||
          antecedent.lane[2] != relation || antecedent.lane[3] != before ||
          antecedent.lane[4] != consequent.lane[4] ||
          antecedent.lane[5] == 0u)
        continue;
      return true;
    }
  }
  return false;
}

// Component closure is an address-free intervention law, not a source-count
// threshold. Every positive fragment in the maximal matching component must be
// removable while another complete cross-Record path still carries the same
// local consequence. The amount of redundancy is therefore grown history,
// while a fragment or bridge that monopolizes the relation cannot authorize it.
BCC32_PARTICIPATION_HD inline bool component_survives_single_record_cuts(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after) {
  bool saw_fragment = false;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) || record.lane[2] != relation ||
        record.lane[5] == 0u)
      continue;
    const bool in_component =
        (record.lane[1] == kAntecedentFragment &&
         record.lane[3] == before) ||
        (record.lane[1] == kConsequentFragment && record.lane[3] == after);
    if (!in_component)
      continue;
    saw_fragment = true;
    if (!positive_pair_exists_excluding(
            state, before, relation, after, slot))
      return false;
  }
  return saw_fragment;
}

// A one-record physical lesion may leave one complete path from a component
// that was demonstrably cut-closed immediately before the lesion. The live
// path still supplies every emitted consequence; the displaced fragment is
// consulted only to establish that this is one-cut survival, never as a route
// or value source. A bridge-only component cannot reach this branch because
// removing either of its two fragments leaves no complete live pair.
BCC32_PARTICIPATION_HD inline bool survives_one_relevant_fragment_lesion(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after) {
  if (state == nullptr || state->lesion.count != 1u)
    return false;
  const rewrite::Record& displaced = state->lesion.displaced[0u];
  if (!is_participation(displaced) || displaced.lane[2] != relation ||
      displaced.lane[5] == 0u)
    return false;
  const bool relevant =
      (displaced.lane[1] == kAntecedentFragment &&
       displaced.lane[3] == before) ||
      (displaced.lane[1] == kConsequentFragment &&
       displaced.lane[3] == after);
  return relevant &&
      positive_pair_exists_excluding(state, before, relation, after,
                                     rewrite::kInvalid);
}

// A cross-context factor is a live relation consequence shared by several
// independently sourced antecedent paths.  It is not a semantic category and
// does not select a fixed number of contexts: the complete resident population
// is walked, every positive source pair is revalidated, and all currently live
// sources participate.  This is the reader needed when two contacts teach the
// same consequence in different contexts; resolve_relation_step deliberately
// remains the stricter reader for one exact antecedent.
BCC32_PARTICIPATION_HD inline bool cross_context_source_pair(
    const rewrite::ResidentRewriteState* state, std::uint32_t source,
    std::uint32_t relation, std::uint32_t after, bool positive,
    std::uint32_t* before) {
  if (state == nullptr || source == 0u || relation == 0u ||
      relation == rewrite::kInvalid || after == 0u ||
      after == rewrite::kInvalid)
    return false;
  const std::uint32_t expected_positive = positive ? 1u : 0u;
  const std::uint32_t expected_counter = positive ? 0u : 1u;
  std::uint32_t antecedent_count = 0u;
  std::uint32_t consequent_count = 0u;
  std::uint32_t total = 0u;
  std::uint32_t antecedent_value = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) || record.lane[4] != source)
      continue;
    ++total;
    if (record.lane[2] != relation || record.lane[5] != expected_positive ||
        record.lane[6] != expected_counter)
      return false;
    if (record.lane[1] == kAntecedentFragment) {
      ++antecedent_count;
      antecedent_value = record.lane[3];
    } else if (record.lane[1] == kConsequentFragment &&
               record.lane[3] == after) {
      ++consequent_count;
    } else {
      return false;
    }
  }
  if (total != 2u || antecedent_count != 1u || consequent_count != 1u)
    return false;
  if (before != nullptr)
    *before = antecedent_value;
  return true;
}

// Reafferent source pairs are still ordinary participation fragments. The
// additional witness check selects only the causal lineage established by an
// accepted physical return; it does not select a returned byte, relation,
// answer, or provider. This keeps an older ordinary-contact closure live and
// competing, while allowing the reafferent receipt to ask the resident
// question it actually owns: did accepted returns form one distributed
// component? A missing, orphaned, or defeated witness cannot participate.
BCC32_PARTICIPATION_HD inline bool cross_context_reafferent_source_pair(
    const rewrite::ResidentRewriteState* state, std::uint32_t source,
    std::uint32_t relation, std::uint32_t after, bool positive,
    std::uint32_t* before) {
  return source_witness::is_reafferent_witness(state, source) &&
      cross_context_source_pair(
          state, source, relation, after, positive, before);
}

BCC32_PARTICIPATION_HD inline bool
cross_context_reafferent_positive_pair_exists_excluding(
    const rewrite::ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after, std::uint32_t excluded_slot) {
  if (state == nullptr || relation == 0u || after == 0u)
    return false;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  for (std::uint32_t consequent_slot = 0u; consequent_slot < capacity;
       ++consequent_slot) {
    if (consequent_slot == excluded_slot)
      continue;
    const rewrite::Record& consequent = state->records[consequent_slot];
    if (!is_participation(consequent) ||
        consequent.lane[1] != kConsequentFragment ||
        consequent.lane[2] != relation || consequent.lane[3] != after ||
        consequent.lane[5] == 0u || consequent.lane[6] != 0u)
      continue;
    if (!cross_context_reafferent_source_pair(
            state, consequent.lane[4], relation, after, true, nullptr))
      continue;
    // cross_context_reafferent_source_pair already rederives the one exact
    // antecedent companion for this source. No second scan may broaden the
    // closure with an unrelated ordinary-contact fragment.
    return true;
  }
  return false;
}

BCC32_PARTICIPATION_HD inline bool
cross_context_component_survives_single_record_cuts(
    const rewrite::ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after) {
  if (state == nullptr || relation == 0u || after == 0u)
    return false;
  bool saw_fragment = false;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) || record.lane[2] != relation ||
        record.lane[5] == 0u || record.lane[6] != 0u ||
        !cross_context_reafferent_source_pair(
            state, record.lane[4], relation, after, true, nullptr))
      continue;
    if ((record.lane[1] == kConsequentFragment && record.lane[3] != after) ||
        (record.lane[1] != kAntecedentFragment &&
         record.lane[1] != kConsequentFragment))
      continue;
    saw_fragment = true;
    if (!cross_context_reafferent_positive_pair_exists_excluding(
            state, relation, after, slot))
      return false;
  }
  return saw_fragment;
}

// Re-derive a relation consequence across all live antecedent contexts. This
// is deliberately separate from ordinary exact-context readout: a single
// context must still fail the cut-closure law, while a consequence repeated
// through distinct antecedent endpoints may form a distributed factor. The
// caller supplies only the relation; candidate consequences and support are
// derived from resident matter. A returned external word is therefore an
// observation of this closure, never its selector.
// Candidate and source uniqueness are re-derived by resident scans rather
// than capacity-sized device allocations. This keeps authority in the live
// records and removes an allocator-dependent failure mode under load.
BCC32_PARTICIPATION_HD inline bool
cross_context_reafferent_after_seen_before(
    const rewrite::ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after, std::uint32_t current_slot) {
  for (std::uint32_t slot = 0u; slot < current_slot; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) ||
        record.lane[1] != kConsequentFragment ||
        record.lane[2] != relation || record.lane[3] != after ||
        record.lane[5] == 0u || record.lane[6] != 0u ||
        !cross_context_reafferent_source_pair(
            state, record.lane[4], relation, after, true, nullptr))
      continue;
    return true;
  }
  return false;
}

BCC32_PARTICIPATION_HD inline bool
cross_context_reafferent_source_seen_before(
    const rewrite::ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after, std::uint32_t source, std::uint32_t current_slot) {
  for (std::uint32_t slot = 0u; slot < current_slot; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_participation(record) ||
        record.lane[1] != kConsequentFragment ||
        record.lane[2] != relation || record.lane[3] != after ||
        record.lane[4] != source || record.lane[5] == 0u ||
        record.lane[6] != 0u ||
        !cross_context_reafferent_source_pair(
            state, source, relation, after, true, nullptr))
      continue;
    return true;
  }
  return false;
}

BCC32_PARTICIPATION_HD inline StepResolution
resolve_cross_context_reafferent_component_from_relation(
    const rewrite::ResidentRewriteState* state, std::uint32_t relation) {
  StepResolution result{};
  if (state == nullptr || relation == 0u || relation == rewrite::kInvalid)
    return result;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  std::uint32_t qualified = 0u;
  for (std::uint32_t candidate_slot = 0u; candidate_slot < capacity;
       ++candidate_slot) {
    const rewrite::Record& candidate_record = state->records[candidate_slot];
    if (!is_participation(candidate_record) ||
        candidate_record.lane[1] != kConsequentFragment ||
        candidate_record.lane[2] != relation ||
        candidate_record.lane[5] == 0u || candidate_record.lane[6] != 0u ||
        !cross_context_reafferent_source_pair(
            state, candidate_record.lane[4], relation,
            candidate_record.lane[3], true, nullptr) ||
        cross_context_reafferent_after_seen_before(
            state, relation, candidate_record.lane[3], candidate_slot))
      continue;

    const std::uint32_t after = candidate_record.lane[3];
    std::uint32_t source_count = 0u;
    std::uint32_t first_before = 0u;
    bool diverse_before = false;
    std::uint32_t latest_revision = 0u;
    for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
      const rewrite::Record& record = state->records[slot];
      if (!is_participation(record) ||
          record.lane[1] != kConsequentFragment ||
          record.lane[2] != relation || record.lane[3] != after ||
          record.lane[5] == 0u || record.lane[6] != 0u)
        continue;
      std::uint32_t before = 0u;
      if (!cross_context_reafferent_source_pair(
              state, record.lane[4], relation, after, true, &before) ||
          cross_context_reafferent_source_seen_before(
              state, relation, after, record.lane[4], slot))
        continue;
      ++source_count;
      if (source_count == 1u)
        first_before = before;
      else if (before != first_before)
        diverse_before = true;
      if (record.revision > latest_revision)
        latest_revision = record.revision;
    }

    std::uint32_t counter = 0u;
    for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
      const rewrite::Record& record = state->records[slot];
      if (is_participation(record) &&
          record.lane[1] == kConsequentFragment &&
          record.lane[2] == relation && record.lane[3] == after &&
          record.lane[5] == 0u && record.lane[6] == 1u &&
          cross_context_reafferent_source_pair(
              state, record.lane[4], relation, after, false, nullptr))
        ++counter;
    }
    const bool closed = diverse_before && counter == 0u &&
        cross_context_component_survives_single_record_cuts(
            state, relation, after);
    if (!closed)
      continue;

    ++qualified;
    if (qualified > 1u) {
      result = StepResolution{};
      result.ambiguous = 1u;
      break;
    }
    result.after = after;
    result.participating_records = source_count * 2u;
    result.independent_sources = source_count;
    result.positive_support = source_count;
    result.counter_support = counter;
    result.consequence_revision = latest_revision;
  }
  if (qualified == 1u)
    result.ready = 1u;
  return result;
}

// Resolve one relation transition only from live resident fragments. The
// returned support counts are observer measurements; authority is the
// cut-closure property and unique live consequence, never a fixed cardinality.
BCC32_PARTICIPATION_HD inline StepResolution resolve_relation_step(
    const rewrite::ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation) {
  StepResolution result{};
  if (state == nullptr || before == 0u || relation == 0u ||
      before == rewrite::kInvalid || relation == rewrite::kInvalid)
    return result;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  std::uint32_t qualified = 0u;
  // Each candidate is the first live record for its alternative. Unique
  // sources are similarly re-derived from earlier live records; no
  // capacity-sized heap workspace is needed for a resident relation step.
  for (std::uint32_t candidate_slot = 0u; candidate_slot < capacity;
       ++candidate_slot) {
    const rewrite::Record& candidate_record = state->records[candidate_slot];
    if (!is_participation(candidate_record) ||
        candidate_record.lane[1] != kConsequentFragment ||
        candidate_record.lane[2] != relation ||
        (!paired_antecedent_exists(state, candidate_record, before, true) &&
         !paired_antecedent_exists(state, candidate_record, before, false)))
      continue;
    const std::uint32_t alternative = candidate_record.lane[3];
    if (relation_alternative_seen_before(state, before, relation, alternative,
                                         candidate_slot))
      continue;
    std::uint32_t positive = 0u;
    std::uint32_t counter = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t records = 0u;
    std::uint32_t latest_revision = 0u;
    for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
      const rewrite::Record& record = state->records[slot];
      if (!is_participation(record) ||
          record.lane[1] != kConsequentFragment ||
          record.lane[2] != relation ||
          record.lane[3] != alternative)
        continue;
      const bool positive_pair = record.lane[5] != 0u &&
          paired_antecedent_exists(state, record, before, true);
      const bool counter_pair = record.lane[6] != 0u &&
          paired_antecedent_exists(state, record, before, false);
      if (!positive_pair && !counter_pair)
        continue;
      positive += positive_pair ? 1u : 0u;
      counter += counter_pair ? 1u : 0u;
      records += 2u;
      if (positive_pair &&
          !relation_positive_source_seen_before(
              state, before, relation, alternative, record.lane[4], slot))
        ++source_count;
      if (positive_pair && record.lane[4] > latest_revision)
        latest_revision = record.lane[4];
    }
    const bool cut_closed = component_survives_single_record_cuts(
        state, before, relation, alternative);
    const bool one_cut_survivor = !cut_closed &&
        survives_one_relevant_fragment_lesion(
            state, before, relation, alternative);
    if (positive == 0u || counter != 0u ||
        (!cut_closed && !one_cut_survivor))
      continue;
    ++qualified;
    if (qualified > 1u) {
      result = StepResolution{};
      result.ambiguous = 1u;
      return result;
    }
    result.after = alternative;
    result.participating_records = records;
    result.independent_sources = source_count;
    result.positive_support = positive;
    result.counter_support = counter;
    result.consequence_revision = latest_revision;
  }
  if (qualified != 1u) {
    result = StepResolution{};
    result.ambiguous = qualified > 1u ? 1u : 0u;
    return result;
  }
  result.ready = 1u;
  return result;
}

// The probe supplies only ordinary observed event matter: one starting
// distinction followed by one to three relation distinctions. It supplies no
// candidate, expected consequence, Record locus, semantic identity, or route.
BCC32_PARTICIPATION_HD inline ProbeReceipt propose_next_event(
    const rewrite::ResidentRewriteState* state,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count) {
  ProbeReceipt receipt{};
  receipt.resident_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || probe == nullptr || probe_count < 2u ||
      probe[0].valid == 0u)
    return receipt;
  receipt.steps_requested = probe_count - 1u;
  std::uint32_t current = probe[0].value;
  receipt.weakest_step_support = 0xffffffffu;
  for (std::uint32_t step = 1u; step < probe_count; ++step) {
    if (probe[step].valid == 0u ||
        probe[step].reserved != rewrite::kEventFrameNone)
      return ProbeReceipt{};
    const StepResolution selected =
        resolve_relation_step(state, current, probe[step].value);
    if (selected.ready == 0u) {
      receipt.ambiguous = selected.ambiguous;
      receipt.proposed_event = 0u;
      return receipt;
    }
    current = selected.after;
    ++receipt.steps_completed;
    receipt.participating_records += selected.participating_records;
    receipt.independent_sources += selected.independent_sources;
    receipt.positive_support += selected.positive_support;
    receipt.counter_support += selected.counter_support;
    if (selected.independent_sources < receipt.weakest_step_support)
      receipt.weakest_step_support = selected.independent_sources;
    if (selected.consequence_revision > receipt.consequence_revision)
      receipt.consequence_revision = selected.consequence_revision;
  }
  receipt.proposed_event = current;
  receipt.ready = receipt.steps_completed == receipt.steps_requested ? 1u : 0u;
  return receipt;
}

// Read every suffix of the one-clock yielded trajectory through physical page
// access. No packet or relation-count limit is imposed: the greatest live
// unambiguous suffix wins, as it did in the bounded precursor, while physical
// record exhaustion remains the only aperture.
BCC32_PARTICIPATION_HD inline SpanReaderContribution
read_current_span_contribution(const rewrite::ResidentRewriteState* state) {
  SpanReaderContribution contribution{};
  contribution.resident_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || state->fault != 0u)
    return contribution;
  const std::uint32_t trajectory_slot =
      rewrite::find_current_trajectory(state);
  if (trajectory_slot == rewrite::kInvalid)
    return contribution;
  const rewrite::Record& trajectory = state->records[trajectory_slot];
  if (trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != rewrite::kFormTrajectory ||
      trajectory.lane[1] == 0u || trajectory.lane[1] == rewrite::kInvalid ||
      trajectory.lane[2] < 2u || trajectory.lane[3] != 0u ||
      (trajectory.lane[7] & ~rewrite::kTrajectoryWasYielded) != 0u ||
      trajectory.reserved[0] != 0u || trajectory.reserved[1] != 0u)
    return contribution;

  for (std::uint32_t start = 0u; start + 1u < trajectory.lane[2]; ++start) {
    SpanReaderContribution candidate{};
    candidate.extent = trajectory.lane[2] - start;
    candidate.weakest_step_support = 0xffffffffu;
    candidate.resident_revision = state->revision;
    std::uint32_t current = 0u;
    if (!rewrite::trajectory_word_at(
            state, trajectory.lane[1], start, &current))
      return SpanReaderContribution{};
    bool readable = true;
    for (std::uint32_t index = start + 1u;
         index < trajectory.lane[2]; ++index) {
      std::uint32_t relation = 0u;
      if (!rewrite::trajectory_word_at(
              state, trajectory.lane[1], index, &relation)) {
        readable = false;
        break;
      }
      const StepResolution selected =
          resolve_relation_step(state, current, relation);
      if (selected.ready == 0u) {
        readable = false;
        candidate.ambiguous = selected.ambiguous;
        break;
      }
      current = selected.after;
      candidate.participating_records += selected.participating_records;
      candidate.independent_sources += selected.independent_sources;
      candidate.positive_support += selected.positive_support;
      if (selected.independent_sources < candidate.weakest_step_support)
        candidate.weakest_step_support = selected.independent_sources;
    }
    if (!readable) {
      if (candidate.ambiguous != 0u) {
        contribution = SpanReaderContribution{};
        contribution.ambiguous = 1u;
        contribution.extent = candidate.extent;
        contribution.resident_revision = state->revision;
        return contribution;
      }
      continue;
    }
    candidate.ready = 1u;
    candidate.word = current;
    // Starts are visited from the maximum physical extent downward. The
    // first ready result is therefore the only candidate that can win; no
    // shorter suffix can change the selected structure.
    contribution = candidate;
    return contribution;
  }
  return contribution;
}

BCC32_PARTICIPATION_HD inline std::uint32_t withdraw_source(
    rewrite::ResidentRewriteState* state, std::uint32_t source_revision) {
  if (state == nullptr || source_revision == 0u)
    return 0u;
  std::uint32_t withdrawn = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    rewrite::Record& record = state->records[slot];
    if (!is_participation(record) || record.lane[4] != source_revision)
      continue;
    rewrite::clear_record(&record);
    ++withdrawn;
    ++state->revision;
  }
  return withdrawn;
}

BCC32_PARTICIPATION_HD inline std::uint32_t participation_count(
    const rewrite::ResidentRewriteState* state) {
  if (state == nullptr)
    return 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot)
    count += is_participation(state->records[slot]) ? 1u : 0u;
  return count;
}

BCC32_PARTICIPATION_HD inline std::uint32_t participation_slot_at(
    const rewrite::ResidentRewriteState* state, std::uint32_t ordinal) {
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    if (!is_participation(state->records[slot]))
      continue;
    if (ordinal == 0u)
      return slot;
    --ordinal;
  }
  return rewrite::kInvalid;
}

BCC32_PARTICIPATION_HD inline DamageReceipt damage_participation(
    rewrite::ResidentRewriteState* state, std::uint32_t count,
    bool dispersed) {
  DamageReceipt receipt{};
  receipt.requested = count;
  receipt.dispersed = dispersed ? 1u : 0u;
  if (state == nullptr || count == 0u || count > 8u ||
      state->lesion.count != 0u)
    return receipt;
  const std::uint32_t total = participation_count(state);
  std::uint32_t selected_slots[8]{};
  std::uint32_t selected_sources[8]{};
  for (std::uint32_t index = 0u; index < count && index < total; ++index) {
    std::uint32_t ordinal = dispersed ? (index * 5u) % total : index;
    std::uint32_t slot = participation_slot_at(state, ordinal);
    for (std::uint32_t retry = 0u; retry < total; ++retry) {
      bool used = false;
      for (std::uint32_t prior = 0u; prior < index; ++prior)
        used |= selected_slots[prior] == slot;
      // A dispersed lesion crosses independently lived source pairs rather
      // than striking both halves of the same pair. It remains a physical
      // damage geometry: source identity is never consulted by learning or
      // proposal, only by this observer-selected perturbation schedule.
      if (dispersed && slot != rewrite::kInvalid) {
        for (std::uint32_t prior = 0u; prior < index; ++prior)
          used |= selected_sources[prior] == state->records[slot].lane[4];
      }
      if (!used)
        break;
      ordinal = (ordinal + 1u) % total;
      slot = participation_slot_at(state, ordinal);
    }
    if (slot == rewrite::kInvalid)
      break;
    selected_slots[index] = slot;
    rewrite::Record& record = state->records[slot];
    selected_sources[index] = record.lane[4];
    const std::uint32_t escrow = state->lesion.count++;
    state->lesion.displaced[escrow] = record;
    state->lesion.original_slot[escrow] = slot;
    state->lesion.removed_matter_q8 += record.matter_q8;
    receipt.removed_matter_q8 += record.matter_q8;
    record = rewrite::Record{};
    record.matter_q8 = 0u;
    ++receipt.displaced;
    ++state->revision;
  }
  return receipt;
}

// Return physical matter, not its learned content. Ordinary contacts must
// rebuild every erased constraint after this repair boundary.
BCC32_PARTICIPATION_HD inline std::uint32_t release_damaged_matter(
    rewrite::ResidentRewriteState* state) {
  if (state == nullptr)
    return 0u;
  const std::uint32_t count = state->lesion.count;
  for (std::uint32_t index = 0u; index < count; ++index) {
    const std::uint32_t slot = state->lesion.original_slot[index];
    if (slot >= rewrite::live_record_capacity(state) ||
        state->records[slot].matter_q8 != 0u)
      return 0u;
  }
  for (std::uint32_t index = 0u; index < count; ++index) {
    const std::uint32_t slot = state->lesion.original_slot[index];
    state->records[slot] = rewrite::Record{};
    state->records[slot].lane[0] = rewrite::kFormEmpty;
    state->records[slot].matter_q8 =
        state->lesion.displaced[index].matter_q8;
  }
  state->lesion = rewrite::LesionEscrow{};
  return count;
}

#include "bcc32_resident_causal_constraint_participation_device_adapters.inl"

}  // namespace substrate::bcc32::resident_causal_constraint_participation

#undef BCC32_PARTICIPATION_HD
