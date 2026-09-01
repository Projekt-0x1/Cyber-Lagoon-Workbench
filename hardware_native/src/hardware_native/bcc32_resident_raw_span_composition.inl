#pragma once

// Bounded, read-only SpanProgram composition compatibility receipt.
//
// Include after causal_rewrite_program_induction.inl.  This is not an
// executor, learner, source-contact receipt, output path, word boundary, or
// claim that arbitrary prose has been learned.  It only compares already
// authoritative resident Records without changing them.

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_RAW_SPAN_HD __host__ __device__
#else
#define BCC32_RAW_SPAN_HD
#endif

#if !defined(BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE)
namespace substrate::bcc32::causal_rewrite {
#endif

struct RawSpanCompositionCandidate {
  std::uint32_t predicted_word = 0u;
  // Slots remain physical lesion/provenance locations.  Identities, not slots,
  // decide agreement; they are never ProgramCandidateConsensus loci.
  std::uint32_t upper_slot = kInvalid;
  std::uint32_t lower_slot = kInvalid;
  std::uint32_t upper_identity = kInvalid;
  std::uint32_t lower_identity = kInvalid;
  std::uint32_t upper_start = 0u;
  std::uint32_t lower_start = 0u;
  bool have_candidate = false;
  bool conflict = false;
};

enum class RawSpanNextStatus : std::uint32_t {
  kNone = 0u,
  kConcrete = 1u,
  kAmbiguous = 2u,
};

// The existing matcher accepts the first owner-bound term it finds.  Refuse a
// malformed owner group here: exactly one header and one term at every ordinal
// must reproduce the canonical induction digest before it can be a witness.
BCC32_RAW_SPAN_HD inline bool raw_span_program_preflight(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
      program.lane[1] == 0u || program.lane[2] == 0u ||
      program.lane[2] > kMaximumSpanProgramTerms || program.lane[4] == 0u ||
      program.lane[4] > kMaximumProgramVariables)
    return false;

  std::uint32_t term_count[kMaximumSpanProgramTerms]{};
  bool variable_seen[kMaximumProgramVariables]{};
  std::uint32_t header_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[1] != program.lane[1]) continue;
    if (record.lane[0] == kFormSpanProgram) {
      ++header_count;
      continue;
    }
    if (record.lane[0] != kFormSpanProgramTerm) continue;
    if (record.lane[2] >= program.lane[2]) return false;
    ++term_count[record.lane[2]];
  }
  if (header_count != 1u) return false;

  std::uint32_t canonical =
      rewrite_mix(kFormSpanProgram, program.lane[2], program.lane[4]);
  for (std::uint32_t ordinal = 0u; ordinal < program.lane[2]; ++ordinal) {
    if (term_count[ordinal] != 1u) return false;
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t channel = 0u;
    std::uint32_t reserved = 0u;
    if (!span_program_term_at(state, program.lane[1], ordinal, &kind, &value,
                              &channel, &reserved) ||
        reserved != 0u || (channel & ~kRawChannelMask) != 0u)
      return false;
    if (kind == kSpanTermLiteral) {
      if ((value & kRawChannelMask) != channel) return false;
    } else if (kind == kSpanTermVariable && value < program.lane[4]) {
      variable_seen[value] = true;
    } else {
      return false;
    }
    canonical = rewrite_mix(canonical, kind, value ^ channel);
  }
  for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable)
    if (!variable_seen[variable]) return false;
  return canonical == program.lane[5];
}

// Canonical lane[5] is derived from ordered owner-independent term structure;
// preflight above verifies it rather than trusting a slot/allocation order.
BCC32_RAW_SPAN_HD inline std::uint32_t raw_span_program_identity(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (!raw_span_program_preflight(state, program_slot)) return kInvalid;
  return state->records[program_slot].lane[5];
}

BCC32_RAW_SPAN_HD inline RawSpanNextStatus raw_span_next_word_at_zero(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t* word) {
  if (state == nullptr || word == nullptr) return RawSpanNextStatus::kNone;
  std::uint32_t bindings[kMaximumProgramVariables]
                        [kMaximumVariableSpanEvents]{};
  std::uint32_t lengths[kMaximumProgramVariables]{};
  std::uint32_t next_term = kInvalid;
  std::uint32_t next_offset = 0u;
  bool next_unbound = false;
  bool complete = false;
  bool ambiguous = false;
  const std::uint32_t match = span_match_prefix_at(
      state, program, trajectory, 0u, &next_term, &next_offset,
      &next_unbound, &complete, &ambiguous, bindings, lengths);
  if (match == kSpanMatchAmbiguous || ambiguous)
    return RawSpanNextStatus::kAmbiguous;
  if (match != kSpanMatchPrefix || next_unbound || complete ||
      next_term == kInvalid)
    return RawSpanNextStatus::kNone;

  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_channel = 0u;
  std::uint32_t ignored_reserved = 0u;
  if (!span_program_term_at(state, program.lane[1], next_term, &kind, &value,
                            &ignored_channel, &ignored_reserved))
    return RawSpanNextStatus::kNone;
  if (kind == kSpanTermLiteral) {
    *word = value;
    return RawSpanNextStatus::kConcrete;
  }
  if (kind != kSpanTermVariable || value >= kMaximumProgramVariables ||
      lengths[value] == 0u || next_offset >= lengths[value])
    return RawSpanNextStatus::kNone;
  *word = bindings[value][next_offset];
  return RawSpanNextStatus::kConcrete;
}

BCC32_RAW_SPAN_HD inline bool raw_span_accepts_first_word(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t word) {
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_channel = 0u;
  std::uint32_t ignored_reserved = 0u;
  return span_program_term_at(state, program.lane[1], 0u, &kind, &value,
                              &ignored_channel, &ignored_reserved) &&
         kind == kSpanTermLiteral && value == word;
}

BCC32_RAW_SPAN_HD inline bool raw_span_locus_before(
    std::uint32_t left_upper, std::uint32_t left_lower,
    std::uint32_t right_upper, std::uint32_t right_lower) {
  return left_upper < right_upper ||
         (left_upper == right_upper && left_lower < right_lower);
}

// Equal identities select a reproducible physical witness only for lesion
// provenance.  A different structural composition always abstains.
BCC32_RAW_SPAN_HD inline void merge_raw_span_composition_candidate(
    RawSpanCompositionCandidate* receipt, std::uint32_t word,
    std::uint32_t upper_identity, std::uint32_t lower_identity,
    std::uint32_t upper_slot, std::uint32_t lower_slot) {
  if (receipt == nullptr) return;
  if (!receipt->have_candidate) {
    receipt->predicted_word = word;
    receipt->upper_slot = upper_slot;
    receipt->lower_slot = lower_slot;
    receipt->upper_identity = upper_identity;
    receipt->lower_identity = lower_identity;
    receipt->upper_start = 0u;
    receipt->lower_start = 0u;  // virtual first lower term, never appended
    receipt->have_candidate = true;
    return;
  }
  if (receipt->predicted_word != word ||
      receipt->upper_identity != upper_identity ||
      receipt->lower_identity != lower_identity) {
    receipt->conflict = true;
  } else if (raw_span_locus_before(upper_slot, lower_slot, receipt->upper_slot,
                                   receipt->lower_slot)) {
    receipt->upper_slot = upper_slot;
    receipt->lower_slot = lower_slot;
  }
}

// Pristine external trajectory only: ordinary generated/cursor continuation is
// intentionally left to the existing executor and ProgramCandidateConsensus.
// The matcher reads every observed word through trajectory_word_at(), so a
// physical page is never a semantic admission or collection bound.
BCC32_RAW_SPAN_HD inline RawSpanCompositionCandidate
collect_raw_span_composition_candidate(const ResidentRewriteState* state,
                                       const Record& trajectory) {
  RawSpanCompositionCandidate receipt{};
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] == 0u ||
      trajectory.lane[3] != 0u || trajectory.lane[7] != 0u)
    return receipt;

  for (std::uint32_t upper_slot = 0u; upper_slot < live_record_capacity(state);
       ++upper_slot) {
    const Record& upper = state->records[upper_slot];
    if (upper.matter_q8 == 0u || upper.lane[0] != kFormSpanProgram ||
        !resident_program_authoritative(state, upper_slot))
      continue;
    const std::uint32_t upper_identity =
        raw_span_program_identity(state, upper_slot);
    if (upper_identity == kInvalid) continue;
    std::uint32_t word = 0u;
    const RawSpanNextStatus next =
        raw_span_next_word_at_zero(state, upper, trajectory, &word);
    if (next == RawSpanNextStatus::kAmbiguous) {
      receipt.conflict = true;
      continue;
    }
    if (next != RawSpanNextStatus::kConcrete) continue;

    for (std::uint32_t lower_slot = 0u; lower_slot < live_record_capacity(state);
         ++lower_slot) {
      if (lower_slot == upper_slot) continue;
      const Record& lower = state->records[lower_slot];
      if (lower.matter_q8 == 0u || lower.lane[0] != kFormSpanProgram ||
          !resident_program_authoritative(state, lower_slot))
        continue;
      const std::uint32_t lower_identity =
          raw_span_program_identity(state, lower_slot);
      if (lower_identity == kInvalid || lower_identity == upper_identity ||
          !raw_span_accepts_first_word(state, lower, word))
        continue;
      merge_raw_span_composition_candidate(&receipt, word, upper_identity,
                                           lower_identity, upper_slot,
                                           lower_slot);
    }
  }
  if (receipt.conflict) receipt.have_candidate = false;
  return receipt;
}

BCC32_RAW_SPAN_HD inline bool raw_span_composition_uses_locus(
    const RawSpanCompositionCandidate& receipt, std::uint32_t slot) {
  return receipt.have_candidate && !receipt.conflict &&
         (receipt.upper_slot == slot || receipt.lower_slot == slot);
}

#if !defined(BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE)
}  // namespace substrate::bcc32::causal_rewrite
#endif

#undef BCC32_RAW_SPAN_HD
