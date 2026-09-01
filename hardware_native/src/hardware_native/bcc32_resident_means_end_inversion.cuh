#pragma once

#include "causal_rewrite_universe.cuh"

namespace substrate::bcc32::causal_rewrite {

// A generated resident consequence can be read backwards through the final
// concrete predecessor/outcome pair of an ordinary mature Program. Earlier
// terms may contain the variables that made the Program learnable from lived
// histories; the selected pair itself must be literal resident matter. This is
// a generic rewrite over executable matter and never creates or supports one.
__host__ __device__ inline bool advance_resident_means_end_inversion_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u ||
      state->generated_word_valid == 0u ||
      state->generated_locus == kInvalid)
    return false;

  const std::uint32_t target = state->generated_word;
  const std::uint32_t target_program = state->generated_locus;
  std::uint32_t predecessor = 0u;
  std::uint32_t witness_locus = kInvalid;
  bool have_witness = false;
  bool conflict = false;

  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (slot == target_program || program.matter_q8 == 0u ||
        program.lane[0] != kFormProgram ||
        !resident_program_authoritative(state, slot) ||
        program.lane[2] < 2u)
      continue;

    std::uint32_t first = 0u;
    std::uint32_t first_meta = 0u;
    std::uint32_t outcome = 0u;
    std::uint32_t outcome_meta = 0u;
    const std::uint32_t predecessor_index = program.lane[2] - 2u;
    const std::uint32_t outcome_index = program.lane[2] - 1u;
    if (!program_term_at(state, program.lane[1], predecessor_index, &first,
                         &first_meta) ||
        !program_term_at(state, program.lane[1], outcome_index, &outcome,
                         &outcome_meta) ||
        first_meta != 0u || outcome_meta != 0u || outcome != target)
      continue;

    if (!have_witness) {
      predecessor = first;
      witness_locus = slot;
      have_witness = true;
    } else if (predecessor != first) {
      conflict = true;
    } else if (slot < witness_locus) {
      // Agreement must not make record allocation order into a chooser.
      witness_locus = slot;
    }
  }

  // Absence is not ambiguity. With no inverse witness, preserve the original
  // generated consequence so installing this regime cannot suppress unrelated
  // language or action. Different predecessors are a real causal conflict and
  // therefore abstain instead of selecting by Record allocation order.
  if (!have_witness) return false;
  if (conflict) {
    clear_generated_word(state);
    return false;
  }

  state->generated_word = predecessor;
  state->generated_word_valid = 1u;
  state->generated_locus = witness_locus;
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite
