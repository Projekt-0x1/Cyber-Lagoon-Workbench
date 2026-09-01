#pragma once

#include "causal_rewrite_universe.cuh"

#if defined(__CUDACC__)
#define BCC32_EPISODIC_HD __host__ __device__
#else
#define BCC32_EPISODIC_HD
#endif

namespace substrate::bcc32::causal_rewrite {

// This is a transient guard, not another resident object.  A mature fixed or
// span program owns every matching cue, including a cue for which its next
// value is unbound, ambiguous, or otherwise abstaining.  RWR4 must not route
// around that abstraction by consulting an exemplar.
BCC32_EPISODIC_HD inline bool mature_program_engaged(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.lane[2] == 0u) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (!resident_program_authoritative(state, slot)) continue;
    if (program.lane[0] == kFormProgram) {
      std::uint32_t next_word = 0u;
      bool unbound = false;
      if (match_program_prefix(state, program, trajectory, &next_word,
                               &unbound) ||
          unbound)
        return true;
    } else if (program.lane[0] == kFormSpanProgram) {
      std::uint32_t bindings[kMaximumProgramVariables]
                            [kMaximumVariableSpanEvents]{};
      std::uint32_t lengths[kMaximumProgramVariables]{};
      std::uint32_t next_term = kInvalid;
      std::uint32_t next_offset = 0u;
      bool next_unbound = false;
      bool complete = false;
      bool ambiguous = false;
      const std::uint32_t match = span_match_prefix(
          state, program, trajectory, &next_term, &next_offset,
          &next_unbound, &complete, &ambiguous, bindings, lengths);
      if (ambiguous || match != kSpanMatchNone) return true;
    }
  }
  return false;
}

// Return true only for an exact physical prefix.  In particular, this does
// not compare channels modulo payload, normalize words, or search a suffix.
BCC32_EPISODIC_HD inline bool retained_prefix_matches(
    const ResidentRewriteState* state, const Record& source,
    const Record& cue) {
  if (source.lane[2] < cue.lane[2] || cue.lane[2] == 0u) return false;
  for (std::uint32_t index = 0u; index < cue.lane[2]; ++index) {
    std::uint32_t source_word = 0u;
    std::uint32_t cue_word = 0u;
    if (!trajectory_word_at(state, source.lane[1], index, &source_word) ||
        !trajectory_word_at(state, cue.lane[1], index, &cue_word) ||
        source_word != cue_word)
      return false;
  }
  return true;
}

// Temporary migration fallback for pre-RWR11 retained episodic matter.
// RWR11 exact Programs own new complete episodes; this scanner remains only
// until every historical retained source has been translated and the
// language/mixed-provenance parity matrix stays green without it.
BCC32_EPISODIC_HD inline bool advance_resident_episodic_completion_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  clear_generated_word(state);

  const std::uint32_t cue_slot = find_current_trajectory(state);
  if (cue_slot == kInvalid) return false;
  const Record& cue = state->records[cue_slot];
  if (cue.lane[2] == 0u || cue.lane[4] == 0u ||
      (cue.lane[7] &
       ~(kTrajectoryHasGenerated | kTrajectoryWasYielded)) != 0u)
    return false;
  if (mature_program_engaged(state, cue)) return false;

  bool saw_matching_episode = false;
  bool saw_episode_end = false;
  bool saw_word = false;
  bool word_conflict = false;
  std::uint32_t candidate = 0u;
  std::uint32_t diagnostic_locus = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& source = state->records[slot];
    if (source.matter_q8 == 0u || source.lane[0] != kFormTrajectory ||
        source.lane[3] == 0u || source.lane[7] != 0u)
      continue;
    if (!retained_prefix_matches(state, source, cue)) continue;
    saw_matching_episode = true;
    if (source.lane[2] == cue.lane[2]) {
      saw_episode_end = true;
      continue;
    }

    std::uint32_t next_word = 0u;
    if (!trajectory_word_at(state, source.lane[1], cue.lane[2], &next_word)) {
      saw_episode_end = true;
      continue;
    }
    const std::uint32_t term_slot = find_owned_block(
        state, kFormTrajectoryTerm, source.lane[1], cue.lane[2] / 2u);
    if (!saw_word) {
      candidate = next_word;
      saw_word = true;
      diagnostic_locus = term_slot;
    } else if (candidate != next_word) {
      word_conflict = true;
    } else if (term_slot != kInvalid &&
               (diagnostic_locus == kInvalid || term_slot < diagnostic_locus)) {
      // Physical provenance is stable receipt data only; it never ranks a
      // candidate or breaks a semantic tie.
      diagnostic_locus = term_slot;
    }
  }

  // Silence covers blank cues, no source, a remembered end, and every
  // disagreement. End-vs-word is intentionally the same abstention as
  // word-vs-word disagreement.
  if (!saw_matching_episode || !saw_word || saw_episode_end || word_conflict)
    return false;
  if (!append_trajectory_word(state, candidate, true)) return false;

  // Use the same generated-word publication rail as the mature Program path;
  // this migration fallback owns no raw-motor writer or alternate action path.
  state->generated_word = candidate;
  state->generated_word_valid = 1u;
  state->generated_locus = diagnostic_locus;
  state->active_locus = diagnostic_locus;
  refresh_receipt(state);
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite

#undef BCC32_EPISODIC_HD
