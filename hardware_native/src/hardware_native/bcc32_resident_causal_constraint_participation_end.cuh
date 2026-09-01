#pragma once

#include <cstdint>

#include "bcc32_resident_causal_constraint_participation.cuh"
#include "bcc32_resident_causal_relation_source_witness.cuh"
#include "bcc32_resident_resource_attention_discourse.cuh"

#if defined(__CUDACC__)
#define BCC32_PARTICIPATION_END_HD __host__ __device__
#else
#define BCC32_PARTICIPATION_END_HD
#endif

namespace substrate::bcc32::resident_causal_constraint_participation::
physical_end {

namespace participation =
    substrate::bcc32::resident_causal_constraint_participation;
namespace rewrite = substrate::bcc32::causal_rewrite;
namespace source_witness = rewrite::resident_causal_relation_source_witness;
namespace resource_attention =
    substrate::bcc32::resident_resource_attention_discourse;

// This receipt is staged resident evidence, not public output. A future RWR0
// caller must invoke it before an END closer retires the current trajectory and
// must itself preserve whole-state atomicity. No raw triplet, relation label,
// source owner, or expected consequence crosses this interface.
struct StageReceipt {
  std::uint32_t attempted = 0u;
  std::uint32_t external_formation = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t factor_intersections = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t progressed = 0u;
  std::uint32_t completed = 0u;
  std::uint32_t windows = 0u;
  // Production matter-paid evidence for the window-admission stride below
  // (0X1-207): resident association capacity immediately before/after this
  // call's windows attempt, and whether a genuine allocator-pressure
  // rejection was resolved by retiring an existing association this call.
  // Staged resident evidence only, same as every other field here.
  resource_attention::ResidentAssociationCapacityReceipt
      association_capacity_before{};
  resource_attention::ResidentAssociationCapacityReceipt
      association_capacity_after{};
  std::uint32_t pressure_turnover = 0u;
  std::uint64_t resident_revision = 0u;
};

inline constexpr std::uint32_t kExternalRelationWorkPerEpoch = 16u;

// Begin only records the physical source identity. It deliberately does not
// call tagged_history(): that helper scans the complete trajectory before the
// resumable 16-event cursor can start. Exact per-word lineage is validated by
// advance_current_before_end, so END admission remains bounded for long raw
// contact.
BCC32_PARTICIPATION_END_HD inline StageReceipt stage_current_before_end(
    rewrite::ResidentRewriteState* state) {
  StageReceipt receipt{};
  receipt.resident_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || state->fault != 0u)
    return receipt;
  receipt.attempted = 1u;
  if (state->external_relation_stage_active != 0u) {
    receipt.rejected = 1u;
    return receipt;
  }
  const std::uint32_t source_slot = rewrite::find_current_trajectory(state);
  if (source_slot == rewrite::kInvalid) {
    receipt.rejected = 1u;
    return receipt;
  }
  const rewrite::Record& source = state->records[source_slot];
  if (source.matter_q8 == 0u ||
      source.lane[0] != rewrite::kFormTrajectory || source.lane[2] == 0u ||
      source.lane[2] < 3u || source.lane[1] == 0u ||
      source.lane[1] == rewrite::kInvalid || source.lane[3] != 0u ||
      source.lane[4] != 0u || source.lane[5] != rewrite::kInvalid ||
      source.lane[7] != 0u) {
    receipt.rejected = 1u;
    return receipt;
  }

  state->external_relation_stage_active = 1u;
  state->external_relation_stage_owner = source.lane[1];
  // Establish the physical root-owner uniqueness proof once, at stage
  // admission.  Later bounded epochs must revalidate this exact locus, not
  // rescan the whole resident ecology merely to rediscover it.
  std::uint32_t root_owner_count = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& candidate = state->records[slot];
    if (candidate.matter_q8 != 0u &&
        candidate.lane[0] == rewrite::kFormTrajectory &&
        candidate.lane[1] == source.lane[1])
      ++root_owner_count;
  }
  if (root_owner_count != 1u) {
    state->external_relation_stage_active = 0u;
    state->external_relation_stage_rejected = 1u;
    receipt.rejected = 1u;
    return receipt;
  }
  state->external_relation_stage_source_slot = source_slot;
  state->external_relation_stage_source_revision = source.revision;
  state->external_relation_stage_event_cursor = 0u;
  state->external_relation_stage_window_cursor = 0u;
  state->external_relation_stage_provenance_ordinal = rewrite::kInvalid;
  state->external_relation_stage_provenance_slot = rewrite::kInvalid;
  state->external_relation_stage_page = rewrite::kInvalid;
  state->external_relation_stage_page_slot = rewrite::kInvalid;
  state->external_relation_stage_term_page = rewrite::kInvalid;
  state->external_relation_stage_term_ordinal = rewrite::kInvalid;
  state->external_relation_stage_term_slot = rewrite::kInvalid;
  state->external_relation_stage_term_alt_page = rewrite::kInvalid;
  state->external_relation_stage_term_alt_ordinal = rewrite::kInvalid;
  state->external_relation_stage_term_alt_slot = rewrite::kInvalid;
  state->external_relation_stage_external_leaves = 0u;
  state->external_relation_stage_admitted = 0u;
  state->external_relation_stage_rejected = 0u;
  state->external_relation_stage_completed = 0u;
  state->external_relation_stage_digest = source_witness::mix(
      source_witness::mix(0x13198a2e03707344ull, source.lane[1]),
      source.revision);
  receipt.progressed = 1u;
  receipt.resident_revision = state->revision;
  return receipt;
}

// Resume the same raw source through a bounded amount of resident work. The
// event cursor validates external lineage and derives the compact witness;
// the window cursor then admits overlapping local constraints. Neither cursor
// is a semantic context window or an answer-producing object.
BCC32_PARTICIPATION_END_HD inline StageReceipt advance_current_before_end(
    rewrite::ResidentRewriteState* state,
    std::uint32_t work_limit = kExternalRelationWorkPerEpoch,
    bool measure_factor_intersections = true) {
  StageReceipt receipt{};
  receipt.resident_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || state->fault != 0u || work_limit == 0u)
    return receipt;
  if (state->external_relation_stage_active == 0u) {
    receipt.completed = 1u;
    return receipt;
  }
  receipt.attempted = 1u;
  const std::uint32_t source_slot =
      state->external_relation_stage_source_slot;
  if (source_slot == rewrite::kInvalid ||
      source_slot >= rewrite::live_record_capacity(state)) {
    state->external_relation_stage_active = 0u;
    state->external_relation_stage_rejected = 1u;
    receipt.rejected = 1u;
    return receipt;
  }
  const rewrite::Record& source = state->records[source_slot];
  if (source.matter_q8 == 0u || source.lane[0] != rewrite::kFormTrajectory ||
      source.lane[1] != state->external_relation_stage_owner ||
      source.revision != state->external_relation_stage_source_revision ||
      source.lane[2] < 3u || source.lane[3] != 0u) {
    state->external_relation_stage_active = 0u;
    state->external_relation_stage_rejected = 1u;
    receipt.rejected = 1u;
    return receipt;
  }

  std::uint32_t work = 0u;
  while (state->external_relation_stage_event_cursor < source.lane[2] &&
         work < work_limit) {
    const std::uint32_t index = state->external_relation_stage_event_cursor;
    const std::uint32_t provenance_ordinal = index / 2u;
    std::uint32_t provenance_slot =
        state->external_relation_stage_provenance_slot;
    if (state->external_relation_stage_provenance_ordinal !=
            provenance_ordinal ||
        provenance_slot == rewrite::kInvalid) {
      provenance_slot = rewrite::find_owned_block(
          state, rewrite::kFormTrajectoryProvenance, source.lane[1],
          provenance_ordinal);
      state->external_relation_stage_provenance_ordinal = provenance_ordinal;
      state->external_relation_stage_provenance_slot = provenance_slot;
    }
    if (provenance_slot == rewrite::kInvalid) {
      state->external_relation_stage_active = 0u;
      state->external_relation_stage_rejected = 1u;
      receipt.rejected = 1u;
      return receipt;
    }
    const rewrite::Record& provenance = state->records[provenance_slot];
    const std::uint32_t local = index % 2u;
    const std::uint32_t origin_lane = 3u + local * 2u;
    const std::uint32_t producer_lane = origin_lane + 1u;
    // `origin_at()` performed the same provenance lookup immediately before
    // this block. Fuse its external-only validation here so one source event
    // pays for one resident block search, not two, without relaxing any
    // lineage condition accepted by the old path.
    if (provenance.matter_q8 == 0u ||
        provenance.lane[0] != rewrite::kFormTrajectoryProvenance ||
        (provenance.lane[rewrite::kProvenanceValidityLane] & ~0x3u) != 0u ||
        (provenance.lane[rewrite::kProvenanceValidityLane] &
         (1u << local)) == 0u ||
        provenance.lane[origin_lane] != rewrite::kProvenanceExternalOrigin ||
        provenance.lane[producer_lane] != rewrite::kInvalid) {
      state->external_relation_stage_active = 0u;
      state->external_relation_stage_rejected = 1u;
      receipt.rejected = 1u;
      return receipt;
    }
    const std::uint32_t page = index / rewrite::kTrajectoryPageEvents;
    const std::uint32_t local_index = index % rewrite::kTrajectoryPageEvents;
    std::uint32_t term_owner = source.lane[1];
    if (page != 0u) {
      std::uint32_t page_slot = state->external_relation_stage_page_slot;
      if (state->external_relation_stage_page != page ||
          page_slot == rewrite::kInvalid) {
        page_slot = rewrite::find_owned_block(
            state, rewrite::kFormTrajectoryPage, source.lane[1], page);
        state->external_relation_stage_page = page;
        state->external_relation_stage_page_slot = page_slot;
      }
      if (page_slot == rewrite::kInvalid) {
        state->external_relation_stage_active = 0u;
        state->external_relation_stage_rejected = 1u;
        receipt.rejected = 1u;
        return receipt;
      }
      const rewrite::Record& continuation = state->records[page_slot];
      if (continuation.lane[3] != page * rewrite::kTrajectoryPageEvents ||
          continuation.lane[4] <= local_index ||
          continuation.lane[4] > rewrite::kTrajectoryPageEvents ||
          continuation.lane[6] == 0u ||
          continuation.lane[6] == rewrite::kInvalid) {
        state->external_relation_stage_active = 0u;
        state->external_relation_stage_rejected = 1u;
        receipt.rejected = 1u;
        return receipt;
      }
      term_owner = continuation.lane[6];
    }
    const std::uint32_t term_ordinal = local_index / 2u;
    std::uint32_t term_slot = state->external_relation_stage_term_slot;
    if (state->external_relation_stage_term_page != page ||
        state->external_relation_stage_term_ordinal != term_ordinal ||
        term_slot == rewrite::kInvalid) {
      term_slot = rewrite::find_owned_block(
          state, rewrite::kFormTrajectoryTerm, term_owner, term_ordinal);
      state->external_relation_stage_term_page = page;
      state->external_relation_stage_term_ordinal = term_ordinal;
      state->external_relation_stage_term_slot = term_slot;
    }
    if (term_slot == rewrite::kInvalid) {
      state->external_relation_stage_active = 0u;
      state->external_relation_stage_rejected = 1u;
      receipt.rejected = 1u;
      return receipt;
    }
    const std::uint32_t value =
        state->records[term_slot].lane[4u + (local_index % 2u)];
    state->external_relation_stage_digest = source_witness::mix(
        state->external_relation_stage_digest, index);
    state->external_relation_stage_digest = source_witness::mix(
        state->external_relation_stage_digest, value);
    state->external_relation_stage_digest = source_witness::mix(
        state->external_relation_stage_digest, provenance.revision);
    state->external_relation_stage_digest = source_witness::mix(
        state->external_relation_stage_digest, provenance.matter_q8);
    ++state->external_relation_stage_event_cursor;
    ++state->external_relation_stage_external_leaves;
    ++work;
    receipt.progressed = 1u;
  }
  if (state->external_relation_stage_event_cursor < source.lane[2]) {
    receipt.resident_revision = state->revision;
    return receipt;
  }

  if (!source_witness::retain_external_source_digest(
          state, source.lane[1], source.revision,
          state->external_relation_stage_digest,
          state->external_relation_stage_external_leaves)) {
    state->external_relation_stage_active = 0u;
    state->external_relation_stage_rejected = 1u;
    receipt.rejected = 1u;
    return receipt;
  }

  if (work < work_limit) {
    std::uint32_t next_window =
        state->external_relation_stage_window_cursor;
    const std::uint32_t prior_window = next_window;
    const resource_attention::MatterPaidAssociationAssimilationReceipt
        paid_assimilation =
            resource_attention::assimilate_matter_paid_external_association_windows(
                state, source_slot, &next_window, work_limit - work, true);
    const participation::AssimilationReceipt& assimilated =
        paid_assimilation.assimilation;
    receipt.association_capacity_before = paid_assimilation.pre_capacity;
    receipt.association_capacity_after = paid_assimilation.post_capacity;
    receipt.pressure_turnover = paid_assimilation.pressure_turnover;
    if (assimilated.rejected != 0u) {
      state->external_relation_stage_active = 0u;
      state->external_relation_stage_rejected = 1u;
      receipt.rejected = 1u;
      return receipt;
    }
    state->external_relation_stage_window_cursor = next_window;
    state->external_relation_stage_admitted += assimilated.admitted;
    receipt.admitted = assimilated.admitted;
    receipt.factor_intersections = assimilated.factor_intersections;
    receipt.windows = next_window;
    receipt.progressed |= assimilated.admitted != 0u ||
                                  next_window != prior_window ||
                                  paid_assimilation.pressure_turnover != 0u
        ? 1u
        : 0u;
  }
  if (state->external_relation_stage_window_cursor <= source.lane[2] - 3u) {
    receipt.resident_revision = state->revision;
    return receipt;
  }

  // Factor formation is observer-only.  The per-window admission path skips
  // its capacity-cubed recount, but preserve the canonical StageReceipt
  // contract by taking one final count after the complete source has been
  // assimilated.  This keeps receipts exact without multiplying the scan by
  // the number of overlapping windows.
  if (measure_factor_intersections)
    receipt.factor_intersections = participation::count_factor_intersections(state);
  state->external_relation_stage_active = 0u;
  state->external_relation_stage_completed = 1u;
  receipt.external_formation = 1u;
  receipt.completed = 1u;
  receipt.windows = state->external_relation_stage_window_cursor;
  receipt.resident_revision = state->revision;
  return receipt;
}

}  // namespace substrate::bcc32::resident_causal_constraint_participation::physical_end

#undef BCC32_PARTICIPATION_END_HD
