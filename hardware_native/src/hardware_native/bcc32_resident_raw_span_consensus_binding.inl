#pragma once

// Bind the bounded raw SpanProgram composition receipt to the canonical
// ProgramCandidateConsensus shape.  Include this after
// causal_rewrite_universe.cuh and bcc32_resident_raw_span_composition.inl.
//
// Binding is read-only and chooses no host label or score.  Public preparation
// re-derives the unique current composition from the pristine trajectory so a
// caller cannot forge a receipt around two valid program identities and an
// arbitrary word.  The emitter then uses the canonical trajectory append path.

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_RAW_SPAN_BIND_HD __host__ __device__
#else
#define BCC32_RAW_SPAN_BIND_HD
#endif

#if !defined(BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE)
namespace substrate::bcc32::causal_rewrite {
#endif

struct RawSpanConsensusBindingReceipt {
  std::uint32_t predicted_word = 0u;
  std::uint32_t upper_slot = kInvalid;
  std::uint32_t lower_slot = kInvalid;
  std::uint32_t upper_identity = kInvalid;
  std::uint32_t lower_identity = kInvalid;
  bool bound = false;
  bool conflict = false;
  bool public_authorized = false;
};

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_unique_trajectory_topology(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || state->fault != 0u ||
      trajectory.lane[1] == 0u ||
      trajectory.lane[1] == kInvalid || trajectory.lane[2] == 0u ||
      trajectory.lane[2] / kTrajectoryPageEvents >= live_record_capacity(state) ||
      trajectory.reserved[0] != 0u || trajectory.reserved[1] != 0u)
    return false;
  std::uint32_t current_headers = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[0] == kFormTrajectory &&
        record.lane[3] == 0u) {
      ++current_headers;
      if (&record != &trajectory) return false;
    }
  }
  if (current_headers != 1u) return false;
  const std::uint32_t page_count =
      (trajectory.lane[2] + kTrajectoryPageEvents - 1u) /
      kTrajectoryPageEvents;
  for (std::uint32_t page = 1u; page < page_count; ++page) {
    const std::uint32_t page_slot =
        trajectory_page_slot(state, trajectory.lane[1], page);
    if (page_slot == kInvalid) return false;
    const Record& continuation = state->records[page_slot];
    const std::uint32_t base = page * kTrajectoryPageEvents;
    const std::uint32_t expected =
        trajectory.lane[2] - base > kTrajectoryPageEvents
            ? kTrajectoryPageEvents
            : trajectory.lane[2] - base;
    if (continuation.lane[3] != base || continuation.lane[4] != expected ||
        continuation.lane[4] == 0u ||
        continuation.lane[6] !=
            trajectory_page_owner(state, trajectory.lane[1], page))
      return false;
    std::uint32_t page_digest = 0u;
    for (std::uint32_t offset = 0u; offset < expected; ++offset) {
      std::uint32_t word = 0u;
      if (!trajectory_word_at(state, trajectory.lane[1], base + offset,
                              &word))
        return false;
      page_digest = rewrite_mix(page_digest, word, base + offset);
    }
    if (continuation.lane[5] != page_digest) return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == kFormTrajectoryPage &&
        record.lane[1] == trajectory.lane[1] &&
        record.lane[2] >= page_count)
      return false;
  }
  for (std::uint32_t page = 0u; page < page_count; ++page) {
    const std::uint32_t base = page * kTrajectoryPageEvents;
    const std::uint32_t extent =
        trajectory.lane[2] - base > kTrajectoryPageEvents
            ? kTrajectoryPageEvents
            : trajectory.lane[2] - base;
    const std::uint32_t blocks = (extent + 1u) / 2u;
    const std::uint32_t term_owner =
        trajectory_page_owner(state, trajectory.lane[1], page);
    for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
      std::uint32_t matches = 0u;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& record = state->records[slot];
        if (record.matter_q8 != 0u &&
            record.lane[0] == kFormTrajectoryTerm &&
            record.lane[1] == term_owner && record.lane[2] == ordinal) {
          if (record.lane[3] != 0u || record.lane[6] != 0u ||
              record.lane[7] != 0u || record.reserved[0] != 0u ||
              record.reserved[1] != 0u)
            return false;
          ++matches;
        }
      }
      if (matches != 1u) return false;
    }
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormTrajectoryTerm)
      continue;
    bool expected = false;
    for (std::uint32_t page = 0u; page < page_count; ++page) {
      const std::uint32_t base = page * kTrajectoryPageEvents;
      const std::uint32_t extent =
          trajectory.lane[2] - base > kTrajectoryPageEvents
              ? kTrajectoryPageEvents
              : trajectory.lane[2] - base;
      if (record.lane[1] == trajectory_page_owner(state, trajectory.lane[1], page) &&
          record.lane[2] < (extent + 1u) / 2u)
        expected = true;
    }
    if (!expected) return false;
  }
  std::uint32_t rolling = 0u;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], index, &word))
      return false;
    rolling = rewrite_mix(rolling, word, index);
  }
  return trajectory.lane[6] == rolling;
}

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_append_capacity(
    const ResidentRewriteState* state, const Record& trajectory) {
  if ((trajectory.lane[2] & 1u) != 0u) return true;
  const std::uint32_t required =
      trajectory.lane[2] != 0u &&
              trajectory.lane[2] % kTrajectoryPageEvents == 0u
          ? 2u
          : 1u;
  std::uint32_t free = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    if (state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormEmpty)
      ++free;
  return free >= required;
}

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_program_exactly_equal(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot) {
  if (!raw_span_program_preflight(state, left_slot) ||
      !raw_span_program_preflight(state, right_slot))
    return false;
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  if (left.lane[2] != right.lane[2] || left.lane[4] != right.lane[4])
    return false;
  for (std::uint32_t ordinal = 0u; ordinal < left.lane[2]; ++ordinal) {
    std::uint32_t left_kind = 0u, left_value = 0u, left_channel = 0u,
                  left_reserved = 0u;
    std::uint32_t right_kind = 0u, right_value = 0u, right_channel = 0u,
                  right_reserved = 0u;
    if (!span_program_term_at(state, left.lane[1], ordinal, &left_kind,
                              &left_value, &left_channel, &left_reserved) ||
        !span_program_term_at(state, right.lane[1], ordinal, &right_kind,
                              &right_value, &right_channel, &right_reserved) ||
        left_kind != right_kind || left_value != right_value ||
        left_channel != right_channel || left_reserved != right_reserved)
      return false;
  }
  return true;
}

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_identity_collision_free(
    const ResidentRewriteState* state, std::uint32_t selected_slot,
    std::uint32_t identity) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
        program.lane[5] != identity || slot == selected_slot)
      continue;
    if (!raw_span_program_exactly_equal(state, selected_slot, slot))
      return false;
  }
  return true;
}

// The canonical trajectory header is the provenance/cursor authority.  Only
// a yielded, untouched external trajectory beginning at offset zero may enter
// this bridge; generated, carried, retained, or cursor-backed matter is not
// fresh evidence for public composition. Before the first generated word the
// canonical generated offset is kInvalid.
BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_pristine_external(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] == 0u ||
      trajectory.lane[3] != 0u || trajectory.lane[4] == 0u ||
      trajectory.lane[5] != kInvalid ||
      trajectory.lane[7] != kTrajectoryWasYielded)
    return false;
  if (!raw_span_binding_unique_trajectory_topology(state, trajectory))
    return false;
  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid || &state->records[current_slot] != &trajectory)
    return false;
  // A live cursor is generated continuation state, even if its trajectory
  // header has not yet acquired the generated flag.
  return find_span_execution_cursor(state, trajectory.lane[1]) == kInvalid;
}

// Composition discovery itself remains defined over pristine pre-yield
// external shape. Public binding occurs after the physical yield boundary, so
// normalize only a local header copy; owned trajectory terms and resident
// program authority remain read from the live state.
BCC32_RAW_SPAN_BIND_HD inline RawSpanCompositionCandidate
collect_raw_span_yielded_binding_candidate(const ResidentRewriteState* state,
                                           const Record& trajectory) {
  if (!raw_span_binding_pristine_external(state, trajectory)) return {};
  Record pristine = trajectory;
  pristine.lane[4] = 0u;
  pristine.lane[7] = 0u;
  return collect_raw_span_composition_candidate(state, pristine);
}

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_authoritative_pair(
    const ResidentRewriteState* state,
    const RawSpanCompositionCandidate& candidate) {
  if (state == nullptr || !candidate.have_candidate || candidate.conflict ||
      candidate.upper_slot >= live_record_capacity(state) ||
      candidate.lower_slot >= live_record_capacity(state) ||
      candidate.upper_slot == candidate.lower_slot ||
      candidate.upper_identity == kInvalid || candidate.lower_identity == kInvalid ||
      candidate.upper_identity == candidate.lower_identity)
    return false;
  if (!resident_program_authoritative(state, candidate.upper_slot) ||
      !resident_program_authoritative(state, candidate.lower_slot))
    return false;
  return raw_span_program_identity(state, candidate.upper_slot) ==
             candidate.upper_identity &&
         raw_span_program_identity(state, candidate.lower_slot) ==
             candidate.lower_identity &&
         raw_span_binding_identity_collision_free(
             state, candidate.upper_slot, candidate.upper_identity) &&
         raw_span_binding_identity_collision_free(
             state, candidate.lower_slot, candidate.lower_identity);
}

BCC32_RAW_SPAN_BIND_HD inline bool raw_span_binding_matches_current_candidate(
    const ResidentRewriteState* state, const Record& trajectory,
    const RawSpanConsensusBindingReceipt& binding) {
  if (!raw_span_binding_pristine_external(state, trajectory)) return false;
  const RawSpanCompositionCandidate current =
      collect_raw_span_yielded_binding_candidate(state, trajectory);
  return current.have_candidate && !current.conflict &&
         current.predicted_word == binding.predicted_word &&
         current.upper_slot == binding.upper_slot &&
         current.lower_slot == binding.lower_slot &&
         current.upper_identity == binding.upper_identity &&
         current.lower_identity == binding.lower_identity;
}

// Merge one already-collected composition receipt into the canonical
// consensus.  The extent and word must agree exactly with any prior canonical
// witness; a different logical composition is an abstention, never a slot- or
// traversal-order choice.  The support argument is deliberately zero: this
// bridge carries no score and does not rank candidates.
BCC32_RAW_SPAN_BIND_HD inline bool bind_raw_span_composition_consensus(
    const ResidentRewriteState* state, const Record& trajectory,
    const RawSpanCompositionCandidate& candidate,
    ProgramCandidateConsensus* consensus,
    RawSpanConsensusBindingReceipt* binding) {
  if (consensus == nullptr || binding == nullptr ||
      !raw_span_binding_pristine_external(state, trajectory) ||
      !raw_span_binding_authoritative_pair(state, candidate)) {
    if (consensus != nullptr) consensus->conflict = true;
    if (binding != nullptr) binding->conflict = true;
    return false;
  }

  const RawSpanCompositionCandidate current =
      collect_raw_span_yielded_binding_candidate(state, trajectory);
  if (!current.have_candidate || current.conflict ||
      current.predicted_word != candidate.predicted_word ||
      current.upper_slot != candidate.upper_slot ||
      current.lower_slot != candidate.lower_slot ||
      current.upper_identity != candidate.upper_identity ||
      current.lower_identity != candidate.lower_identity) {
    consensus->conflict = true;
    binding->conflict = true;
    return false;
  }

  if (binding->bound &&
      (binding->predicted_word != candidate.predicted_word ||
       binding->upper_identity != candidate.upper_identity ||
       binding->lower_identity != candidate.lower_identity)) {
    consensus->conflict = true;
    binding->conflict = true;
    return false;
  }
  if (consensus->conflict ||
      (consensus->have_candidate &&
       (consensus->word != candidate.predicted_word ||
        consensus->extent != trajectory.lane[2]))) {
    consensus->conflict = true;
    binding->conflict = true;
    return false;
  }

  merge_program_candidate(consensus, candidate.predicted_word,
                          candidate.upper_slot, true, trajectory.lane[2], 0u);
  if (consensus->conflict) {
    binding->conflict = true;
    return false;
  }

  binding->predicted_word = candidate.predicted_word;
  binding->upper_identity = candidate.upper_identity;
  binding->lower_identity = candidate.lower_identity;
  binding->upper_slot = candidate.upper_slot;
  binding->lower_slot = candidate.lower_slot;
  binding->bound = true;
  binding->public_authorized = false;
  return true;
}

// Prepare the public-emitter handoff.  The caller must
// pass the same canonical consensus that drove the prospective output.  No
// generated/carry/lesioned trajectory can reach this receipt; the physical
// yielded boundary is mandatory, and a conflicted or logically different
// consensus remains silent.
BCC32_RAW_SPAN_BIND_HD inline bool prepare_raw_span_public_emission(
    const ResidentRewriteState* state, const Record& trajectory,
    const ProgramCandidateConsensus& consensus,
    RawSpanConsensusBindingReceipt* binding) {
  if (binding == nullptr || !binding->bound || binding->conflict ||
      !raw_span_binding_matches_current_candidate(state, trajectory, *binding) ||
      !raw_span_binding_authoritative_pair(
          state, RawSpanCompositionCandidate{
                     binding->predicted_word, binding->upper_slot,
                     binding->lower_slot, binding->upper_identity,
                     binding->lower_identity, 0u, 0u, true, false}) ||
      !consensus.have_candidate || consensus.conflict ||
      !consensus.selected_from_span || !consensus.span_contributed ||
      consensus.word != binding->predicted_word ||
      consensus.extent != trajectory.lane[2] ||
      consensus.diagnostic_locus != binding->upper_slot) {
    if (binding != nullptr) {
      binding->public_authorized = false;
      binding->conflict = true;
    }
    return false;
  }
  binding->public_authorized = true;
  return true;
}

// Publish through the same generated-word state and trajectory mutation used
// by the canonical resident executor.  This operation remains deliberately
// separate from collection: only the exact consensus that passed the
// read-only binding may append one generated word.  A second call sees the
// generated trajectory flag and fails closed.
BCC32_RAW_SPAN_BIND_HD inline bool emit_raw_span_consensus_word(
    ResidentRewriteState* state, Record& trajectory,
    const ProgramCandidateConsensus& consensus,
    RawSpanConsensusBindingReceipt* binding) {
  if (state == nullptr || state->fault != 0u || binding == nullptr ||
      state->generated_word_valid != 0u ||
      trajectory.lane[2] / kTrajectoryPageEvents >= live_record_capacity(state) ||
      !raw_span_binding_append_capacity(state, trajectory) ||
      state->program_generated_events == 0xffffffffu ||
      state->span_generated_events == 0xffffffffu ||
      !prepare_raw_span_public_emission(state, trajectory, consensus, binding)) {
    if (binding != nullptr) {
      binding->public_authorized = false;
      binding->conflict = true;
    }
    return false;
  }
  if (!append_trajectory_word(state, consensus.word, true)) {
    binding->public_authorized = false;
    binding->conflict = true;
    return false;
  }
  state->generated_word = consensus.word;
  state->generated_word_valid = 1u;
  state->generated_locus = binding->upper_slot;
  state->active_locus = binding->upper_slot;
  ++state->program_generated_events;
  ++state->span_generated_events;
  refresh_receipt(state);
  return true;
}

#if !defined(BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE)
}  // namespace substrate::bcc32::causal_rewrite
#endif

#undef BCC32_RAW_SPAN_BIND_HD
