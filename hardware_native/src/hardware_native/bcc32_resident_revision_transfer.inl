// Grounded resident revision transfer.
//
// A mature fixed Program has already earned a variable topology from repeated
// external experience. One later exact Program from an accepted resident-issued
// action return may extend that topology while changing its final predicted
// literal. The product is ordinary executable Program matter; its donor, exact
// source, and ticket-return receipt remain causally load-bearing. Neither
// generated trajectories nor ordinary unticketed teaching can authorize it.

// The rewrite core is included beneath the inquiry extension, so this lower
// layer names the extension's stable Record wire forms without including the
// higher layer back into itself.
inline constexpr std::uint32_t kRevisionInquiryHeaderForm = 0x94c6a217u;
inline constexpr std::uint32_t kRevisionInquiryAlternativeForm = 0xa5d7b328u;
inline constexpr std::uint32_t kRevisionInquiryReplyWitnessForm = 0xc7f9d54au;
inline constexpr std::uint32_t kRevisionInquiryReplyTermForm = 0x6a1d3f90u;
inline constexpr std::uint32_t kRevisionInquiryResumeWitnessForm = 0xd80ae65bu;
inline constexpr std::uint32_t kRevisionInquiryConstructorForm = 0xe91bf76cu;
inline constexpr std::uint32_t kRevisionInquiryConstructorTermForm = 0xfa2c087du;
inline constexpr std::uint32_t kRevisionInquiryConstructorWitnessForm = 0x8b3d198eu;
inline constexpr std::uint32_t kRevisionInquiryRequiredState =
    1u | (1u << 2u) | (1u << 3u) | (1u << 4u) | (1u << 5u) |
    (1u << 6u);
inline constexpr std::uint32_t kRevisionInquiryCapturedState = 1u << 1u;
inline constexpr std::uint32_t kRevisionInquiryExternalWitness = 0x45585431u;

// GitHub #1060 (duplicate #1162 closed onto this ticket) forward
// declaration. `revision_intervention_lineage_authoritative` below used to
// close a real 8-function mutual-recursion cycle by calling back into
// `resident_program_authoritative`
// (bcc32_resident_cross_context_anti_unification.cuh) for each of an
// inquiry's two alternative programs -- and one of those programs can
// itself be a revision-transfer product whose own donor inquiry needs the
// same check again. `748220e41d` threaded a defaulted `recursion_depth`
// parameter through the cycle as a runtime safety net, but its own commit
// message says plainly that nvlink still reports STACK:UNKNOWN with the
// guard in place: nvlink's static stack-bound analysis cannot bound a
// cyclic call graph regardless of any runtime recursion guard, because the
// cycle is still a cycle in the compiled object code. The fix has to
// remove the back edge from the *compiled* call graph, not merely bound it
// at runtime. `revision_lineage_bounded::
// revision_intervention_lineage_authoritative_bounded`, defined at the end
// of this file, is a faithful non-recursive reimplementation of the whole
// reachable closure using an explicit bounded worklist instead of the C++
// call stack; see the comment block above its definition for the full
// cycle map. `recursion_depth` is accepted here only so this function's
// signature still matches the call sites `748220e41d` updated (E and F
// below still pass it); the bounded worklist has its own, independent
// bound and does not need it.
namespace revision_lineage_bounded {
BCC32_CAUSAL_GERMLINE_DISPATCH bool
revision_intervention_lineage_authoritative_bounded(
    const ResidentRewriteState* state, std::uint32_t inquiry_owner,
    std::uint32_t inquiry_revision);
}  // namespace revision_lineage_bounded

BCC32_CAUSAL_GERMLINE_DISPATCH bool
revision_intervention_lineage_authoritative(
    const ResidentRewriteState* state, std::uint32_t inquiry_owner,
    std::uint32_t inquiry_revision, std::uint32_t recursion_depth = 0u) {
  (void)recursion_depth;
  // GitHub #1060: the original recursive body now lives, byte-for-byte
  // equivalent in behavior, as `revision_lineage_bounded::evaluate_inquiry`
  // near the end of this file. Its one recursive dependency (this
  // function's own "is each alternative program authoritative" check) is
  // answered from an explicit bounded worklist there instead of a direct
  // call back into `resident_program_authoritative`, which is what lets
  // nvlink compute a real per-kernel stack bound instead of reporting
  // EIATTR_MIN_STACK_SIZE=0xffffffff / STACK:UNKNOWN.
  return revision_lineage_bounded::
      revision_intervention_lineage_authoritative_bounded(
          state, inquiry_owner, inquiry_revision);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool current_revision_intervention_lineage(
    const ResidentRewriteState* state, std::uint32_t producer_locus,
    std::uint32_t generated_word, std::uint32_t* inquiry_owner,
    std::uint32_t* inquiry_revision) {
  if (inquiry_owner != nullptr) *inquiry_owner = kInvalid;
  if (inquiry_revision != nullptr) *inquiry_revision = 0u;
  if (state == nullptr || producer_locus >= live_record_capacity(state) ||
      inquiry_owner == nullptr || inquiry_revision == nullptr)
    return false;
  if (state->records[producer_locus].matter_q8 == 0u) return false;
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  const Record& trajectory = state->records[trajectory_slot];
  if (trajectory.lane[2] < 2u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u)
    return false;
  std::uint32_t last = 0u;
  if (!trajectory_word_at(state, trajectory.lane[1], trajectory.lane[2] - 1u,
                          &last) ||
      last != generated_word)
    return false;
  const std::uint32_t action_index = trajectory.lane[2] - 1u;
  const std::uint32_t provenance_slot = find_owned_block(
      state, kFormTrajectoryProvenance, trajectory.lane[1], action_index / 2u);
  const std::uint32_t local = action_index % 2u;
  if (provenance_slot == kInvalid ||
      (state->records[provenance_slot].lane[7] & (1u << local)) == 0u ||
      state->records[provenance_slot].lane[3u + local * 2u] != 1u ||
      state->records[provenance_slot].lane[4u + local * 2u] != producer_locus)
    return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& inquiry = state->records[slot];
    if (inquiry.matter_q8 == 0u ||
        inquiry.lane[0] != kRevisionInquiryHeaderForm ||
        inquiry.lane[2] != trajectory.lane[1] ||
        !revision_intervention_lineage_authoritative(
            state, inquiry.lane[1], inquiry.revision))
      continue;
    std::uint32_t selected_word = kInvalid;
    std::uint32_t exact_fork = 0u;
    Record prefix = trajectory;
    prefix.lane[2] = inquiry.lane[3];
    for (std::uint32_t binding_slot = 0u; binding_slot < live_record_capacity(state);
         ++binding_slot) {
      const Record& binding = state->records[binding_slot];
      if (binding.matter_q8 == 0u ||
          binding.lane[0] != kRevisionInquiryAlternativeForm ||
          binding.lane[1] != inquiry.lane[1])
        continue;
      std::uint32_t program_slot = kInvalid;
      for (std::uint32_t candidate = 0u; candidate < live_record_capacity(state);
           ++candidate) {
        const Record& program = state->records[candidate];
        if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
            program.lane[1] != binding.lane[2])
          continue;
        if (program_slot != kInvalid || program.revision != binding.lane[3])
          return false;
        program_slot = candidate;
      }
      if (program_slot == kInvalid) return false;
      const Record& program = state->records[program_slot];
      std::uint32_t consequence = 0u;
      std::uint32_t meta = 0u;
      bool exact = true;
      if ((program.lane[7] & kProgramFlagVersionSpace) != 0u) {
        for (std::uint32_t index = 0u; exact && index < inquiry.lane[3];
             ++index) {
          std::uint32_t expected = 0u;
          std::uint32_t observed = 0u;
          exact = version_space_effective_program_term_at(
                      state, program, index, &expected, &meta) &&
                  meta == 0u &&
                  trajectory_word_at(state, trajectory.lane[1], index,
                                     &observed) &&
                  expected == observed;
        }
        exact = exact && version_space_effective_program_term_at(
                             state, program, inquiry.lane[3], &consequence,
                             &meta) &&
                meta == 0u;
      } else {
        bool unbound = false;
        exact = match_program_prefix(state, program, prefix, &consequence,
                                     &unbound) &&
                !unbound;
      }
      if (!exact || consequence != binding.lane[4]) return false;
      ++exact_fork;
      if (binding.lane[2] == inquiry.lane[5] &&
          binding.lane[3] == inquiry.lane[6])
        selected_word = binding.lane[4];
    }
    std::uint32_t observed = 0u;
    if (found != kInvalid || exact_fork != 2u || selected_word == kInvalid ||
        trajectory.lane[2] <= inquiry.lane[3] + 1u ||
        !trajectory_word_at(state, trajectory.lane[1], inquiry.lane[3],
                            &observed) ||
        observed != selected_word)
      return false;
    found = slot;
  }
  if (found == kInvalid) return false;
  *inquiry_owner = state->records[found].lane[1];
  *inquiry_revision = state->records[found].revision;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_base_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      program.lane[7] != kProgramFlagEnabled ||
      program.lane[2] < 2u ||
      program.lane[3] < kProgramMatureSupport || program.lane[4] == 0u ||
      program.lane[4] > kMaximumProgramVariables)
    return false;
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    std::uint32_t word = 0u;
    std::uint32_t meta = 0u;
    if (!program_term_at(state, program.lane[1], index, &word, &meta) ||
        meta > program.lane[4])
      return false;
  }
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t revision_return_owner(
    std::uint32_t source_owner, std::uint32_t source_revision,
    std::uint32_t return_witness_owner) {
  return rewrite_mix(
      kFormRevisionTransferReturn, source_owner,
      rewrite_mix(source_revision, return_witness_owner,
                  kCausalGermlineExternal));
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t ticketed_return_witness_owner(
    std::uint32_t action_owner, std::uint32_t trajectory_owner) {
  return rewrite_mix(kFormTicketedReturnWitness, action_owner,
                     trajectory_owner);
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t revision_action_owner(
    std::uint32_t producer_locus, std::uint32_t generated_word,
    std::uint32_t producer_revision, std::uint32_t inquiry_owner,
    std::uint32_t inquiry_revision) {
  return rewrite_mix(kFormRevisionActionIssuance, producer_locus,
                     rewrite_mix(generated_word, producer_revision,
                                 rewrite_mix(inquiry_owner, inquiry_revision,
                                             kCausalGermlineExternal)));
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t revision_ambiguity_trace_owner(
    std::uint32_t action_owner, std::uint32_t inquiry_owner) {
  return rewrite_mix(kFormRevisionAmbiguityTrace, action_owner,
                     inquiry_owner);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_ambiguity_trace_authoritative(
    const ResidentRewriteState* state, const Record& issued) {
  if (state == nullptr || issued.lane[0] != kFormRevisionActionIssuance ||
      issued.reserved[1] == 0u || issued.reserved[1] == kInvalid)
    return false;
  std::uint32_t trace_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& trace = state->records[slot];
    if (trace.matter_q8 == 0u || trace.lane[0] != kFormRevisionAmbiguityTrace ||
        trace.lane[1] != issued.reserved[1])
      continue;
    if (trace_slot != kInvalid) return false;
    trace_slot = slot;
  }
  if (trace_slot == kInvalid) return false;
  const Record& trace = state->records[trace_slot];
  if (trace.lane[1] != revision_ambiguity_trace_owner(issued.lane[1],
                                                       issued.lane[6]) ||
      trace.lane[2] != issued.lane[1] || trace.lane[3] != issued.lane[6] ||
      trace.lane[4] != issued.reserved[0] ||
      trace.lane[5] != issued.lane[5] || trace.lane[6] != issued.lane[4] ||
      trace.lane[7] != kCausalGermlineEnabled)
    return false;
  std::uint32_t participants = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < 2u; ++ordinal) {
    std::uint32_t participant_slot = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& participant = state->records[slot];
      if (participant.matter_q8 == 0u ||
          participant.lane[0] != kFormRevisionAmbiguityParticipant ||
          participant.lane[1] != trace.lane[1] ||
          participant.lane[2] != ordinal)
        continue;
      if (participant_slot != kInvalid) return false;
      participant_slot = slot;
    }
    if (participant_slot == kInvalid) return false;
    const Record& participant = state->records[participant_slot];
    std::uint32_t bindings = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& binding = state->records[slot];
      bindings += binding.matter_q8 != 0u &&
                  binding.lane[0] == kRevisionInquiryAlternativeForm &&
                  binding.lane[1] == issued.lane[6] &&
                  binding.lane[2] == participant.lane[3] &&
                  binding.lane[3] == participant.lane[4] &&
                  binding.lane[4] == participant.lane[5];
    }
    if (bindings != 1u || participant.lane[6] != issued.lane[6] ||
        participant.lane[7] != kCausalGermlineEnabled)
      return false;
    ++participants;
  }
  return participants == 2u;
}

#include "bcc32_resident_revision_egress_lineage.inl"

// Capture the generated consequence before transport exposes its ticket.
BCC32_CAUSAL_GERMLINE_DISPATCH bool mark_current_revision_action_issuance(
    ResidentRewriteState* state) {
  if (state == nullptr || state->generated_locus == kInvalid ||
      state->generated_locus >= live_record_capacity(state) ||
      state->generated_word_valid == 0u ||
      (state->generated_word & kRawChannelMask) != (1u << 24u))
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& existing = state->records[slot];
    if (existing.matter_q8 != 0u &&
        existing.lane[0] == kFormRevisionActionIssuance &&
        existing.lane[7] == 0u)
      return false;
  }
  const Record& producer = state->records[state->generated_locus];
  if (producer.matter_q8 == 0u) return false;
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  const Record trajectory = state->records[trajectory_slot];
  std::uint32_t egress_word = 0u;
  if (trajectory.lane[2] == 0u ||
      free_record_count(state) < 5u ||
      !trajectory_word_at(state, trajectory.lane[1],
                          trajectory.lane[2] - 1u, &egress_word) ||
      egress_word != state->generated_word)
    return false;
  std::uint32_t inquiry_owner = kInvalid;
  std::uint32_t inquiry_revision = 0u;
  if (!current_revision_intervention_lineage(
          state, state->generated_locus, state->generated_word,
          &inquiry_owner, &inquiry_revision))
    return false;
  Record alternatives[2]{};
  std::uint32_t alternative_count = 0u;
  for (std::uint32_t candidate = 0u; candidate < live_record_capacity(state);
       ++candidate) {
    const Record& binding = state->records[candidate];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kRevisionInquiryAlternativeForm ||
        binding.lane[1] != inquiry_owner)
      continue;
    if (alternative_count == 2u) return false;
    alternatives[alternative_count++] = binding;
  }
  if (alternative_count != 2u) return false;
  if (alternatives[1].lane[4] < alternatives[0].lane[4] ||
      (alternatives[1].lane[4] == alternatives[0].lane[4] &&
       alternatives[1].lane[2] < alternatives[0].lane[2])) {
    const Record temporary = alternatives[0];
    alternatives[0] = alternatives[1];
    alternatives[1] = temporary;
  }
  const std::uint32_t owner = revision_action_owner(
      state->generated_locus, state->generated_word, producer.revision,
      inquiry_owner, inquiry_revision);
  if (owner == 0u || owner == kInvalid || record_owner_exists(state, owner))
    return false;
  const std::uint32_t trace_owner =
      revision_ambiguity_trace_owner(owner, inquiry_owner);
  if (trace_owner == 0u || trace_owner == kInvalid ||
      record_owner_exists(state, trace_owner))
    return false;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& issued = state->records[slot];
  issued.lane[0] = kFormRevisionActionIssuance;
  issued.lane[1] = owner;
  issued.lane[2] = state->generated_locus;
  issued.lane[3] = state->generated_word;
  issued.lane[4] = producer.revision;
  issued.lane[5] = producer.lane[1];
  issued.lane[6] = inquiry_owner;
  issued.reserved[0] = inquiry_revision;
  issued.reserved[1] = trace_owner;
  ++issued.revision;
  const std::uint32_t egress_slot = allocate_record(state);
  Record& egress = state->records[egress_slot];
  egress.lane[0] = kFormRevisionEgressWitness;
  egress.lane[1] = owner;
  egress.lane[2] = trajectory.lane[1];
  egress.lane[3] = trajectory.revision + 1u;
  egress.lane[4] = trajectory.lane[2];
  egress.lane[5] = state->generated_word;
  egress.lane[6] = state->generated_locus;
  egress.reserved[0] = producer.revision;
  egress.reserved[1] = producer.lane[1];
  ++egress.revision;
  const std::uint32_t trace_slot = allocate_record(state);
  Record& trace = state->records[trace_slot];
  trace.lane[0] = kFormRevisionAmbiguityTrace;
  trace.lane[1] = trace_owner;
  trace.lane[2] = owner;
  trace.lane[3] = inquiry_owner;
  trace.lane[4] = inquiry_revision;
  trace.lane[5] = producer.lane[1];
  trace.lane[6] = producer.revision;
  trace.lane[7] = kCausalGermlineEnabled;
  ++trace.revision;
  for (std::uint32_t ordinal = 0u; ordinal < 2u; ++ordinal) {
    const std::uint32_t participant_slot = allocate_record(state);
    Record& participant = state->records[participant_slot];
    participant.lane[0] = kFormRevisionAmbiguityParticipant;
    participant.lane[1] = trace_owner;
    participant.lane[2] = ordinal;
    participant.lane[3] = alternatives[ordinal].lane[2];
    participant.lane[4] = alternatives[ordinal].lane[3];
    participant.lane[5] = alternatives[ordinal].lane[4];
    participant.lane[6] = inquiry_owner;
    participant.lane[7] = kCausalGermlineEnabled;
    ++participant.revision;
  }
  Record& retained_egress = state->records[trajectory_slot];
  retained_egress.lane[3] = kRevisionEgressRetainedTrajectory;
  ++retained_egress.revision;
  ++state->revision;
  return revision_ambiguity_trace_authoritative(state, issued) &&
         revision_egress_witness_authoritative(state, issued, false);
}

// Called only inside the already validated accepted-return staging world. The
// exact current trajectory is copied into ordinary owner-bound Record matter;
// no scalar digest substitutes for the returned words.
BCC32_CAUSAL_GERMLINE_DISPATCH bool mark_current_accepted_return_trajectory(
    ResidentRewriteState* state) {
  if (state == nullptr) return false;
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  std::uint32_t action_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& issued = state->records[slot];
    if (issued.matter_q8 == 0u ||
        issued.lane[0] != kFormRevisionActionIssuance ||
        issued.lane[7] != 0u)
      continue;
    if (action_slot != kInvalid) return false;
    action_slot = slot;
  }
  // A generic action return remains revision-inert.
  if (action_slot == kInvalid) return true;
  Record& issued = state->records[action_slot];
  if (issued.lane[2] >= live_record_capacity(state)) return false;
  const Record& producer = state->records[issued.lane[2]];
  if (issued.lane[1] != revision_action_owner(
                            issued.lane[2], issued.lane[3],
                            issued.lane[4], issued.lane[6],
                            issued.reserved[0]) ||
      producer.matter_q8 == 0u ||
      producer.revision != issued.lane[4] ||
      producer.lane[1] != issued.lane[5] ||
      !revision_ambiguity_trace_authoritative(state, issued) ||
      !revision_egress_witness_authoritative(state, issued, false) ||
      !revision_intervention_lineage_authoritative(
          state, issued.lane[6], issued.reserved[0]))
    return false;
  const Record& trajectory = state->records[trajectory_slot];
  if (trajectory.lane[2] == 0u || trajectory.lane[7] != 0u ||
      trajectory.lane[1] == 0u || trajectory.lane[1] == kInvalid)
    return false;
  const std::uint32_t owner = ticketed_return_witness_owner(
      issued.lane[1], trajectory.lane[1]);
  if (owner == 0u || owner == kInvalid || record_owner_exists(state, owner))
    return false;
  if (free_record_count(state) < trajectory.lane[2] + 1u) return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t ignored = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], index, &ignored))
      return false;
  }
  const std::uint32_t header_slot = allocate_record(state);
  if (header_slot == kInvalid) return false;
  Record& witness = state->records[header_slot];
  witness.lane[0] = kFormTicketedReturnWitness;
  witness.lane[1] = owner;
  witness.lane[2] = trajectory.lane[1];
  witness.lane[3] = trajectory.revision;
  witness.lane[4] = trajectory.lane[2];
  witness.lane[5] = issued.lane[1];
  witness.lane[6] = issued.lane[6];
  witness.lane[7] = kCausalGermlineExternal;
  witness.reserved[0] = issued.reserved[0];
  ++witness.revision;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    const std::uint32_t term_slot = allocate_record(state);
    if (term_slot == kInvalid) return false;
    Record& term = state->records[term_slot];
    term.lane[0] = kFormTicketedReturnWitnessTerm;
    term.lane[1] = owner;
    term.lane[2] = index;
    (void)trajectory_word_at(state, trajectory.lane[1], index,
                             &term.lane[3]);
    term.lane[4] = trajectory.lane[1];
    term.lane[5] = trajectory.revision;
    term.lane[6] = trajectory.lane[2];
    term.lane[7] = kCausalGermlineExternal;
    ++term.revision;
  }
  issued.lane[7] = kCausalGermlineExternal;
  ++issued.revision;
  Record* egress = revision_egress_witness(state, issued);
  if (egress == nullptr || egress->lane[7] != 0u) return false;
  egress->lane[7] = kCausalGermlineExternal;
  ++egress->revision;
  ++state->revision;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool ticketed_return_witness_authoritative(
    const ResidentRewriteState* state, std::uint32_t witness_slot,
    std::uint32_t source_slot, std::uint32_t recursion_depth = 0u) {
  const bool require_source = source_slot != kInvalid;
  if (state == nullptr || witness_slot >= live_record_capacity(state) ||
      (require_source &&
       (source_slot >= live_record_capacity(state) ||
        !pure_external_exact_program_authoritative(state, source_slot))))
    return false;
  const Record& witness = state->records[witness_slot];
  if (witness.matter_q8 == 0u ||
      witness.lane[0] != kFormTicketedReturnWitness ||
      witness.lane[1] == 0u || witness.lane[1] == kInvalid ||
      witness.lane[4] == 0u || witness.lane[5] == 0u ||
      witness.lane[5] == kInvalid || witness.lane[6] == 0u ||
      witness.lane[6] == kInvalid ||
      witness.lane[7] != kCausalGermlineExternal ||
      witness.reserved[0] == 0u || witness.reserved[1] != 0u ||
      witness.lane[1] != ticketed_return_witness_owner(
                              witness.lane[5], witness.lane[2]) ||
      !revision_intervention_lineage_authoritative(
          state, witness.lane[6], witness.reserved[0], recursion_depth))
    return false;
  std::uint32_t action_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& issued = state->records[slot];
    if (issued.matter_q8 == 0u ||
        issued.lane[0] != kFormRevisionActionIssuance ||
        issued.lane[1] != witness.lane[5])
      continue;
    if (issued.lane[7] != kCausalGermlineExternal ||
        issued.lane[2] >= live_record_capacity(state) ||
        issued.lane[6] != witness.lane[6] ||
        issued.reserved[0] != witness.reserved[0] ||
        !revision_ambiguity_trace_authoritative(state, issued) ||
        !revision_egress_witness_authoritative(state, issued, true) ||
        issued.lane[1] != revision_action_owner(
                              issued.lane[2], issued.lane[3], issued.lane[4],
                              issued.lane[6], issued.reserved[0]) ||
        state->records[issued.lane[2]].matter_q8 == 0u ||
        state->records[issued.lane[2]].revision != issued.lane[4] ||
        state->records[issued.lane[2]].lane[1] != issued.lane[5])
      return false;
    ++action_count;
  }
  if (action_count != 1u) return false;
  const Record* source = require_source ? &state->records[source_slot] : nullptr;
  if (require_source &&
      (witness.lane[2] != source->lane[1] ||
       witness.lane[4] != source->lane[2]))
    return false;
  std::uint32_t term_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u ||
        term.lane[0] != kFormTicketedReturnWitnessTerm ||
        term.lane[1] != witness.lane[1])
      continue;
    if (term.lane[2] >= witness.lane[4] ||
        term.lane[4] != witness.lane[2] ||
        term.lane[5] != witness.lane[3] ||
        term.lane[6] != witness.lane[4] ||
        term.lane[7] != kCausalGermlineExternal || term.revision != 1u)
      return false;
    if (require_source) {
      std::uint32_t source_word = 0u;
      std::uint32_t meta = 0u;
      if (!program_term_at(state, source->lane[1], term.lane[2], &source_word,
                           &meta) ||
          meta != 0u || source_word != term.lane[3])
        return false;
    }
    ++term_count;
  }
  for (std::uint32_t index = 0u; index < witness.lane[4]; ++index) {
    std::uint32_t owned = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      owned += term.matter_q8 != 0u &&
               term.lane[0] == kFormTicketedReturnWitnessTerm &&
               term.lane[1] == witness.lane[1] && term.lane[2] == index;
    }
    if (owned != 1u) return false;
  }
  return term_count == witness.lane[4];
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool ticketed_revision_source_authoritative(
    const ResidentRewriteState* state, std::uint32_t source_slot,
    std::uint32_t recursion_depth = 0u) {
  if (state == nullptr || source_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, source_slot))
    return false;
  const Record& source = state->records[source_slot];
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& returned = state->records[slot];
    if (returned.matter_q8 == 0u ||
        returned.lane[0] != kFormRevisionTransferReturn ||
        returned.lane[3] != source.lane[1])
      continue;
    if (returned.lane[1] == 0u || returned.lane[1] == kInvalid ||
        returned.lane[2] != source.revision ||
        returned.lane[4] != source.lane[5] ||
        returned.lane[5] != source.lane[2] ||
        returned.lane[6] == 0u || returned.lane[6] == kInvalid ||
        returned.lane[7] != kCausalGermlineExternal ||
        returned.lane[1] != revision_return_owner(
                                source.lane[1], source.revision,
                                returned.lane[6]) ||
        returned.reserved[0] != 0u || returned.reserved[1] != 0u)
      return false;
    std::uint32_t witness_count = 0u;
    std::uint32_t action_owner = kInvalid;
    for (std::uint32_t witness_slot = 0u; witness_slot < live_record_capacity(state);
         ++witness_slot) {
      const Record& witness = state->records[witness_slot];
      if (witness.matter_q8 == 0u ||
          witness.lane[0] != kFormTicketedReturnWitness ||
          witness.lane[1] != returned.lane[6])
        continue;
      if (!ticketed_return_witness_authoritative(state, witness_slot,
                                                  source_slot,
                                                  recursion_depth))
        return false;
      action_owner = witness.lane[5];
      ++witness_count;
    }
    if (witness_count != 1u) return false;
    std::uint32_t action_count = 0u;
    for (std::uint32_t action_slot = 0u; action_slot < live_record_capacity(state);
         ++action_slot) {
      const Record& issued = state->records[action_slot];
      if (issued.matter_q8 == 0u ||
          issued.lane[0] != kFormRevisionActionIssuance ||
          issued.lane[1] != action_owner)
        continue;
      if (issued.lane[7] != kCausalGermlineExternal ||
          issued.lane[2] >= live_record_capacity(state) ||
          issued.lane[1] != revision_action_owner(
                                issued.lane[2], issued.lane[3],
                                issued.lane[4], issued.lane[6],
                                issued.reserved[0]) ||
          state->records[issued.lane[2]].matter_q8 == 0u ||
          state->records[issued.lane[2]].revision != issued.lane[4] ||
          state->records[issued.lane[2]].lane[1] != issued.lane[5] ||
          !revision_ambiguity_trace_authoritative(state, issued) ||
          !revision_egress_witness_authoritative(state, issued, true) ||
          !revision_intervention_lineage_authoritative(
              state, issued.lane[6], issued.reserved[0], recursion_depth))
        return false;
      ++action_count;
    }
    if (action_count != 1u) return false;
    ++count;
  }
  return count == 1u;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_source_return_lineage(
    const ResidentRewriteState* state, std::uint32_t source_slot,
    std::uint32_t* p4_owner, std::uint32_t* trace_owner) {
  if (p4_owner == nullptr || trace_owner == nullptr ||
      !ticketed_revision_source_authoritative(state, source_slot))
    return false;
  const Record& source = state->records[source_slot];
  std::uint32_t returned_count = 0u;
  *p4_owner = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& returned = state->records[slot];
    if (returned.matter_q8 != 0u &&
        returned.lane[0] == kFormRevisionTransferReturn &&
        returned.lane[3] == source.lane[1]) {
      *p4_owner = returned.lane[6];
      ++returned_count;
    }
  }
  std::uint32_t action_owner = kInvalid;
  std::uint32_t p4_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& p4 = state->records[slot];
    if (p4.matter_q8 != 0u && p4.lane[0] == kFormTicketedReturnWitness &&
        p4.lane[1] == *p4_owner) {
      action_owner = p4.lane[5];
      ++p4_count;
    }
  }
  std::uint32_t action_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& issued = state->records[slot];
    if (issued.matter_q8 == 0u ||
        issued.lane[0] != kFormRevisionActionIssuance ||
        issued.lane[1] != action_owner)
      continue;
    if (!revision_ambiguity_trace_authoritative(state, issued)) return false;
    *trace_owner = issued.reserved[1];
    ++action_count;
  }
  return returned_count == 1u && p4_count == 1u && action_count == 1u;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_source_shape_relation(
    const ResidentRewriteState* state, std::uint32_t donor_slot,
    std::uint32_t source_slot, std::uint32_t* mismatch_out) {
  if (state == nullptr || mismatch_out == nullptr ||
      !revision_base_program_authoritative(state, donor_slot) ||
      !pure_external_exact_program_authoritative(state, source_slot))
    return false;
  const Record& donor = state->records[donor_slot];
  const Record& source = state->records[source_slot];
  // Equal extent is a normal relation/referent correction. Longer returned
  // continuations revise the remainder of the composed explanation.
  if (source.lane[2] < donor.lane[2]) return false;

  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  std::uint32_t mismatch = kInvalid;
  for (std::uint32_t index = 0u; index < donor.lane[2]; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    std::uint32_t source_meta = 0u;
    if (!program_term_at(state, source.lane[1], index, &observed,
                         &source_meta) ||
        source_meta != 0u ||
        !program_term_at(state, donor.lane[1], index, &expected, &meta))
      return false;
    if (meta == 0u) {
      if (observed == expected) continue;
      if (index + 1u != donor.lane[2] || mismatch != kInvalid ||
          !same_raw_channel(observed, expected))
        return false;
      mismatch = index;
      continue;
    }
    const std::uint32_t variable = meta - 1u;
    if (variable >= donor.lane[4] ||
        !same_raw_channel(observed, expected))
      return false;
    const std::uint32_t bit = 1u << variable;
    if ((bound & bit) != 0u) {
      if (binding[variable] != observed) return false;
    } else {
      binding[variable] = observed;
      bound |= bit;
    }
  }
  if (mismatch == kInvalid) return false;
  *mismatch_out = mismatch;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_source_relation(
    const ResidentRewriteState* state, std::uint32_t donor_slot,
    std::uint32_t source_slot, std::uint32_t* mismatch_out) {
  return ticketed_revision_source_authoritative(state, source_slot) &&
         revision_source_shape_relation(state, donor_slot, source_slot,
                                        mismatch_out);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_participation_reader_term_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta);

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t
unique_revision_reader_reacquisition_candidate(
    const ResidentRewriteState* state, std::uint32_t source_slot) {
  if (state == nullptr || source_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, source_slot))
    return kInvalid;
  const Record& source = state->records[source_slot];
  std::uint32_t selected = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& reader = state->records[slot];
    if (reader.matter_q8 == 0u ||
        reader.lane[0] != kFormRevisionParticipationReader ||
        reader.lane[2] != source.lane[2])
      continue;
    std::uint32_t witnesses = 0u;
    bool old_source_live = false;
    for (std::uint32_t witness_slot = 0u; witness_slot < live_record_capacity(state);
         ++witness_slot) {
      const Record& witness = state->records[witness_slot];
      if (witness.matter_q8 == 0u ||
          witness.lane[0] != kFormRevisionTransferWitness ||
          witness.lane[1] != reader.lane[1])
        continue;
      ++witnesses;
      const std::uint32_t old_source =
          causal_program_slot_by_owner(state, witness.lane[3]);
      old_source_live |= old_source != kInvalid &&
                         ticketed_revision_source_authoritative(state,
                                                                 old_source);
    }
    if (witnesses != 1u || old_source_live) continue;
    bool exact = true;
    for (std::uint32_t index = 0u; exact && index < source.lane[2]; ++index) {
      std::uint32_t reader_word = 0u;
      std::uint32_t reader_meta = 0u;
      std::uint32_t source_word = 0u;
      std::uint32_t source_meta = 0u;
      exact = revision_participation_reader_term_at(
                  state, reader.lane[1], index, &reader_word, &reader_meta) &&
              program_term_at(state, source.lane[1], index, &source_word,
                              &source_meta) &&
              source_meta == 0u && reader_word == source_word;
    }
    if (!exact) continue;
    if (selected != kInvalid) return kInvalid;
    selected = slot;
  }
  return selected;
}

// Called once after END. It resolves the sole persistent P4 witness against
// the Program converted in place from that witness's exact trajectory. The
// witness and every owned word remain load-bearing after acceptance.
BCC32_CAUSAL_GERMLINE_DISPATCH bool accept_ticketed_revision_source(
    ResidentRewriteState* state) {
  if (state == nullptr) return false;
  std::uint32_t witness_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormTicketedReturnWitness)
      continue;
    bool already_consumed = false;
    for (std::uint32_t returned_slot = 0u; returned_slot < live_record_capacity(state);
         ++returned_slot) {
      const Record& returned = state->records[returned_slot];
      already_consumed |= returned.matter_q8 != 0u &&
                          returned.lane[0] == kFormRevisionTransferReturn &&
                          returned.lane[6] == witness.lane[1];
    }
    if (already_consumed) continue;
    if (witness_slot != kInvalid) return false;
    witness_slot = slot;
  }
  if (witness_slot == kInvalid) return true;
  const Record witness = state->records[witness_slot];
  std::uint32_t source_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!pure_external_exact_program_authoritative(state, slot) ||
        state->records[slot].lane[1] != witness.lane[2])
      continue;
    if (source_slot != kInvalid) return false;
    source_slot = slot;
  }
  if (!ticketed_return_witness_authoritative(state, witness_slot,
                                              source_slot))
    return false;
  bool eligible = source_slot != kInvalid &&
                  !ticketed_revision_source_authoritative(state, source_slot);
  std::uint32_t donor_slot = kInvalid;
  for (std::uint32_t slot = 0u; eligible && slot < live_record_capacity(state); ++slot) {
    std::uint32_t mismatch = kInvalid;
    if (!revision_source_shape_relation(state, slot, source_slot, &mismatch))
      continue;
    if (donor_slot != kInvalid) {
      eligible = false;
      break;
    }
    donor_slot = slot;
  }
  const std::uint32_t reacquisition_reader =
      eligible && donor_slot == kInvalid
          ? unique_revision_reader_reacquisition_candidate(state, source_slot)
          : kInvalid;
  eligible &= (donor_slot != kInvalid || reacquisition_reader != kInvalid) &&
              free_record_count(state) != 0u;
  std::uint32_t owner = kInvalid;
  if (eligible) {
    const Record& source = state->records[source_slot];
    owner = revision_return_owner(source.lane[1], source.revision,
                                  witness.lane[1]);
    eligible = owner != 0u && owner != kInvalid &&
               !record_owner_exists(state, owner);
  }

  if (!eligible) {
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      Record& term = state->records[slot];
      if (term.matter_q8 != 0u &&
          term.lane[0] == kFormTicketedReturnWitnessTerm &&
          term.lane[1] == witness.lane[1])
        clear_record(&term);
    }
    clear_record(&state->records[witness_slot]);
    ++state->revision;
    refresh_receipt(state);
    return true;
  }
  const std::uint32_t receipt_slot = allocate_record(state);
  if (receipt_slot == kInvalid) return false;
  Record& returned = state->records[receipt_slot];
  returned.lane[0] = kFormRevisionTransferReturn;
  returned.lane[1] = owner;
  returned.lane[2] = state->records[source_slot].revision;
  returned.lane[3] = state->records[source_slot].lane[1];
  returned.lane[4] = state->records[source_slot].lane[5];
  returned.lane[5] = state->records[source_slot].lane[2];
  returned.lane[6] = witness.lane[1];
  returned.lane[7] = kCausalGermlineExternal;
  ++returned.revision;
  ++state->revision;
  refresh_receipt(state);
  return ticketed_revision_source_authoritative(state, source_slot);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_participation_reader_term_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta) {
  if (state == nullptr || word == nullptr || meta == nullptr) return false;
  std::uint32_t readers = 0u;
  std::uint32_t term_slot = kInvalid;
  std::uint32_t terms = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    readers += record.matter_q8 != 0u &&
               record.lane[0] == kFormRevisionParticipationReader &&
               record.lane[1] == owner;
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormRevisionParticipationReaderTerm ||
        record.lane[1] != owner || record.lane[2] != index / 2u)
      continue;
    term_slot = slot;
    ++terms;
  }
  if (readers != 1u || terms != 1u) return false;
  const std::uint32_t offset = (index % 2u) * 2u;
  *meta = state->records[term_slot].lane[3u + offset];
  *word = state->records[term_slot].lane[4u + offset];
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
resident_revision_participation_reader_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth = 0u) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& product = state->records[program_slot];
  const std::uint32_t flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagRevisionTransferProduct;
  if (product.matter_q8 == 0u ||
      product.lane[0] != kFormRevisionParticipationReader ||
      product.lane[7] != flags || product.lane[2] < 3u ||
      product.lane[3] < kProgramMatureSupport || product.lane[4] == 0u ||
      product.lane[4] > kMaximumProgramVariables)
    return false;

  std::uint32_t witness_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness ||
        witness.lane[1] != product.lane[1])
      continue;
    if (witness_slot != kInvalid) return false;
    witness_slot = slot;
  }
  if (witness_slot == kInvalid) return false;
  const Record& witness = state->records[witness_slot];
  std::uint32_t source_use_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& source_use = state->records[slot];
    if (source_use.matter_q8 == 0u ||
        source_use.lane[0] != kFormRevisionTransferSourceUse ||
        source_use.lane[1] != product.lane[1])
      continue;
    if (source_use_slot != kInvalid) return false;
    source_use_slot = slot;
  }
  if (source_use_slot == kInvalid) return false;
  const Record& source_use = state->records[source_use_slot];
  const std::uint32_t source_slot =
      causal_program_slot_by_owner(state, witness.lane[3]);
  if (source_slot == kInvalid || witness.lane[2] != product.lane[1] ||
      witness.lane[7] != kCausalGermlineExternal ||
      source_use.lane[2] != witness.lane[3] ||
      source_use.lane[3] != witness.lane[5] ||
      source_use.lane[7] != kCausalGermlineExternal ||
      state->records[source_slot].lane[5] != witness.lane[5] ||
      witness.reserved[0] == 0u || witness.reserved[1] < 2u ||
      witness.reserved[1] > product.lane[2] ||
      product.lane[2] != state->records[source_slot].lane[2])
    return false;
  if (!ticketed_revision_source_authoritative(state, source_slot,
                                               recursion_depth))
    return false;
  std::uint32_t p4_owner = kInvalid;
  std::uint32_t trace_owner = kInvalid;
  if (!revision_source_return_lineage(state, source_slot, &p4_owner,
                                      &trace_owner))
    return false;

  std::uint32_t prior_count = 0u;
  std::uint32_t delta_count = 0u;
  std::uint32_t mismatch = kInvalid;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  for (std::uint32_t index = 0u; index < product.lane[2]; ++index) {
    std::uint32_t product_word = 0u;
    std::uint32_t product_meta = 0u;
    std::uint32_t source_word = 0u;
    std::uint32_t source_meta = 0u;
    if (!revision_participation_reader_term_at(
            state, product.lane[1], index, &product_word, &product_meta) ||
        !program_term_at(state, state->records[source_slot].lane[1], index,
                         &source_word, &source_meta) ||
        source_meta != 0u || product_word != source_word)
      return false;
    std::uint32_t expected_meta = 0u;
    std::uint32_t before_word = kInvalid;
    if (index < witness.reserved[1]) {
      std::uint32_t prior_slot = kInvalid;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& prior = state->records[slot];
        if (prior.matter_q8 == 0u ||
            prior.lane[0] != kFormRevisionTransferPriorTerm ||
            prior.lane[1] != product.lane[1] || prior.lane[2] != index)
          continue;
        if (prior_slot != kInvalid || prior.lane[5] != witness.reserved[0] ||
            prior.lane[6] != witness.reserved[1] ||
            prior.lane[7] != kCausalGermlineEnabled)
          return false;
        prior_slot = slot;
      }
      if (prior_slot == kInvalid) return false;
      const Record& prior = state->records[prior_slot];
      before_word = prior.lane[3];
      expected_meta = prior.lane[4];
      if (expected_meta == 0u) {
        if (prior.lane[3] != source_word) {
          if (index + 1u != witness.reserved[1] || mismatch != kInvalid ||
              !same_raw_channel(prior.lane[3], source_word))
            return false;
          mismatch = index;
        }
      } else {
        const std::uint32_t variable = expected_meta - 1u;
        if (variable >= product.lane[4]) return false;
        const std::uint32_t bit = 1u << variable;
        if ((bound & bit) != 0u && binding[variable] != source_word)
          return false;
        binding[variable] = source_word;
        bound |= bit;
      }
      ++prior_count;
    }
    std::uint32_t focal_deltas = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& delta = state->records[slot];
      if (delta.matter_q8 == 0u ||
          delta.lane[0] != kFormRevisionParticipationDelta ||
          delta.lane[1] != product.lane[1] || delta.lane[2] != index)
        continue;
      const bool extension = index >= witness.reserved[1];
      if (++focal_deltas != 1u || delta.lane[3] != before_word ||
          delta.lane[4] != (extension ? kInvalid : expected_meta) ||
          delta.lane[5] != source_word || delta.lane[6] != expected_meta ||
          delta.lane[7] != (extension ? 2u : 1u) ||
          delta.reserved[0] != p4_owner || delta.reserved[1] != trace_owner)
        return false;
    }
    const bool changed = index >= witness.reserved[1] ||
                         before_word != source_word;
    if (focal_deltas != (changed ? 1u : 0u)) return false;
    delta_count += focal_deltas;
    if (expected_meta > product.lane[4])
        return false;
    if (product_meta != expected_meta) return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& prior = state->records[slot];
    if (prior.matter_q8 != 0u &&
        prior.lane[0] == kFormRevisionTransferPriorTerm &&
        prior.lane[1] == product.lane[1] && prior.lane[2] >= witness.reserved[1])
      return false;
    const Record& delta = state->records[slot];
    if (delta.matter_q8 != 0u &&
        delta.lane[0] == kFormRevisionParticipationDelta &&
        delta.lane[1] == product.lane[1] && delta.lane[2] >= product.lane[2])
      return false;
  }
  return prior_count == witness.reserved[1] && mismatch == witness.lane[6] &&
         delta_count != 0u;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
resident_revision_transfer_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth) {
  return resident_revision_participation_reader_authoritative(
      state, program_slot, recursion_depth);
}

// Revision readers do not fall through the generic Program collector. Their
// next public word is rederived directly from the exact participation-backed
// reader while the ordinary current trajectory supplies the partial cue.
BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_participation_reader_prefix_next(
    const ResidentRewriteState* state, const Record& reader,
    const Record& trajectory, std::uint32_t* next_word,
    bool* next_unbound) {
  if (state == nullptr || next_word == nullptr || next_unbound == nullptr ||
      reader.lane[0] != kFormRevisionParticipationReader ||
      trajectory.lane[2] == 0u || trajectory.lane[2] >= reader.lane[2])
    return false;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    std::uint32_t observed = 0u;
    if (!revision_participation_reader_term_at(
            state, reader.lane[1], index, &expected, &meta) ||
        !trajectory_word_at(state, trajectory.lane[1], index, &observed))
      return false;
    if (meta == 0u) {
      if (expected != observed) return false;
      continue;
    }
    const std::uint32_t variable = meta - 1u;
    if (variable >= reader.lane[4] ||
        !same_raw_channel(expected, observed))
      return false;
    const std::uint32_t bit = 1u << variable;
    if ((bound & bit) != 0u && binding[variable] != observed) return false;
    binding[variable] = observed;
    bound |= bit;
  }
  std::uint32_t stored = 0u;
  std::uint32_t meta = 0u;
  if (!revision_participation_reader_term_at(
          state, reader.lane[1], trajectory.lane[2], &stored, &meta))
    return false;
  if (meta == 0u) {
    *next_word = stored;
    *next_unbound = false;
    return true;
  }
  const std::uint32_t variable = meta - 1u;
  if (variable >= reader.lane[4] || (bound & (1u << variable)) == 0u) {
    *next_unbound = true;
    return true;
  }
  *next_word = binding[variable];
  *next_unbound = false;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_participation_reader_engaged(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.lane[2] == 0u) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& reader = state->records[slot];
    if (reader.matter_q8 == 0u ||
        reader.lane[0] != kFormRevisionParticipationReader ||
        !resident_revision_participation_reader_authoritative(state, slot) ||
        trajectory.lane[2] >= reader.lane[2])
      continue;
    std::uint32_t word = 0u;
    bool unbound = false;
    if (revision_participation_reader_prefix_next(
            state, reader, trajectory, &word, &unbound) ||
        unbound)
      return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
advance_resident_revision_participation_reader_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  clear_generated_word(state);
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  Record& trajectory = state->records[trajectory_slot];
  if (trajectory.lane[4] == 0u) return false;
  std::uint32_t selected = kInvalid;
  std::uint32_t selected_word = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& reader = state->records[slot];
    if (reader.matter_q8 == 0u ||
        reader.lane[0] != kFormRevisionParticipationReader ||
        !resident_revision_participation_reader_authoritative(state, slot) ||
        trajectory.lane[2] >= reader.lane[2])
      continue;
    std::uint32_t word = 0u;
    bool unbound = false;
    if (!revision_participation_reader_prefix_next(
            state, reader, trajectory, &word, &unbound) ||
        unbound)
      continue;
    // Even byte-equal close readers are an unresolved physical ambiguity; no
    // allocation-order winner may reach the public writer.
    if (selected != kInvalid) return false;
    selected = slot;
    selected_word = word;
  }
  if (selected == kInvalid) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& distractor = state->records[slot];
    if (slot == selected || distractor.matter_q8 == 0u ||
        distractor.lane[0] != kFormProgram ||
        (distractor.lane[7] & (kProgramFlagVersionSpace |
                               kProgramFlagRevisionTransferProduct)) != 0u ||
        !resident_program_authoritative(state, slot) ||
        trajectory.lane[2] >= distractor.lane[2])
      continue;
    std::uint32_t distractor_word = 0u;
    bool unbound = false;
    if (match_program_prefix(state, distractor, trajectory, &distractor_word,
                             &unbound) &&
        !unbound)
      return false;
  }
  if (!append_trajectory_word(state, selected_word, true))
    return false;
  state->generated_word = selected_word;
  state->generated_word_valid = 1u;
  state->generated_locus = selected;
  state->active_locus = selected;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool resident_program_shadowed_by_revision(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth) {
  if (resident_revision_transfer_product_authoritative(state, program_slot,
                                                        recursion_depth))
    return true;
  if (!revision_base_program_authoritative(state, program_slot)) return false;
  const std::uint32_t owner = state->records[program_slot].lane[1];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness ||
        witness.lane[2] != owner)
      continue;
    const std::uint32_t product_slot =
        causal_program_slot_by_owner(state, witness.lane[1]);
    if (product_slot != kInvalid &&
        resident_revision_transfer_product_authoritative(state, product_slot,
                                                          recursion_depth))
      return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t
unique_revision_product_for_base(const ResidentRewriteState* state,
                                 std::uint32_t base_slot) {
  if (state == nullptr || base_slot >= live_record_capacity(state))
    return kInvalid;
  if (resident_revision_transfer_product_authoritative(state, base_slot))
    return base_slot;
  if (!revision_base_program_authoritative(state, base_slot)) return kInvalid;
  const std::uint32_t base_owner = state->records[base_slot].lane[1];
  std::uint32_t selected = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness ||
        witness.lane[2] != base_owner)
      continue;
    const std::uint32_t product =
        causal_program_slot_by_owner(state, witness.lane[1]);
    if (product == kInvalid ||
        !resident_revision_transfer_product_authoritative(state, product) ||
        (selected != kInvalid && selected != product))
      return kInvalid;
    selected = product;
  }
  return selected;
}

// Once an exact external Program has been consumed as the evidence source for
// a revision product it remains resident evidence, but it must not compete as
// an independent executable Program. Otherwise physical slot order can select
// the teaching episode instead of the revision product carrying donor lineage.
BCC32_CAUSAL_GERMLINE_DISPATCH bool
pure_external_program_consumed_by_revision(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, program_slot))
    return false;
  const Record& source = state->records[program_slot];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& use = state->records[slot];
    if (use.matter_q8 != 0u &&
        use.lane[0] == kFormRevisionTransferSourceUse &&
        use.lane[2] == source.lane[1] && use.lane[3] == source.lane[5] &&
        use.lane[7] == kCausalGermlineExternal)
      return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_participation_relation(
    const ResidentRewriteState* state, std::uint32_t witness_slot,
    std::uint32_t* source_slot_out, std::uint32_t* variables_out) {
  if (source_slot_out != nullptr) *source_slot_out = kInvalid;
  if (variables_out != nullptr) *variables_out = 0u;
  if (state == nullptr || witness_slot >= live_record_capacity(state) ||
      source_slot_out == nullptr || variables_out == nullptr)
    return false;
  const Record& witness = state->records[witness_slot];
  if (witness.matter_q8 == 0u ||
      witness.lane[0] != kFormRevisionTransferWitness ||
      witness.lane[1] == 0u || witness.lane[1] == kInvalid ||
      witness.lane[2] != witness.lane[1] || witness.lane[3] == 0u ||
      witness.lane[3] == kInvalid || witness.lane[7] != kCausalGermlineExternal ||
      witness.reserved[0] == 0u || witness.reserved[1] < 2u)
    return false;
  const std::uint32_t source_slot =
      causal_program_slot_by_owner(state, witness.lane[3]);
  if (source_slot == kInvalid ||
      !ticketed_revision_source_authoritative(state, source_slot))
    return false;
  const Record& source = state->records[source_slot];
  if (source.lane[2] < witness.reserved[1] ||
      source.lane[5] != witness.lane[5])
    return false;
  std::uint32_t p4_owner = kInvalid;
  std::uint32_t trace_owner = kInvalid;
  if (!revision_source_return_lineage(state, source_slot, &p4_owner,
                                      &trace_owner))
    return false;
  std::uint32_t source_uses = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& use = state->records[slot];
    if (use.matter_q8 == 0u ||
        use.lane[0] != kFormRevisionTransferSourceUse ||
        use.lane[1] != witness.lane[1])
      continue;
    if (use.lane[2] != source.lane[1] || use.lane[3] != source.lane[5] ||
        use.lane[7] != kCausalGermlineExternal)
      return false;
    ++source_uses;
  }
  if (source_uses != 1u) return false;

  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  std::uint32_t variables = 0u;
  std::uint32_t mismatch = kInvalid;
  for (std::uint32_t index = 0u; index < witness.reserved[1]; ++index) {
    std::uint32_t prior_slot = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& prior = state->records[slot];
      if (prior.matter_q8 == 0u ||
          prior.lane[0] != kFormRevisionTransferPriorTerm ||
          prior.lane[1] != witness.lane[1] || prior.lane[2] != index)
        continue;
      if (prior_slot != kInvalid || prior.lane[5] != witness.reserved[0] ||
          prior.lane[6] != witness.reserved[1] ||
          prior.lane[7] != kCausalGermlineEnabled)
        return false;
      prior_slot = slot;
    }
    if (prior_slot == kInvalid) return false;
    const Record& prior = state->records[prior_slot];
    std::uint32_t source_word = 0u;
    std::uint32_t source_meta = 0u;
    if (!program_term_at(state, source.lane[1], index, &source_word,
                         &source_meta) ||
        source_meta != 0u)
      return false;
    std::uint32_t deltas = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& delta = state->records[slot];
      if (delta.matter_q8 == 0u ||
          delta.lane[0] != kFormRevisionParticipationDelta ||
          delta.lane[1] != witness.lane[1] || delta.lane[2] != index)
        continue;
      if (++deltas != 1u || delta.lane[3] != prior.lane[3] ||
          delta.lane[4] != prior.lane[4] || delta.lane[5] != source_word ||
          delta.lane[6] != prior.lane[4] || delta.lane[7] != 1u ||
          delta.reserved[0] != p4_owner || delta.reserved[1] != trace_owner)
        return false;
    }
    if (deltas != (prior.lane[3] != source_word ? 1u : 0u)) return false;
    if (prior.lane[4] == 0u) {
      if (prior.lane[3] == source_word) continue;
      if (index + 1u != witness.reserved[1] || mismatch != kInvalid ||
          !same_raw_channel(prior.lane[3], source_word))
        return false;
      mismatch = index;
      continue;
    }
    const std::uint32_t variable = prior.lane[4] - 1u;
    if (variable >= kMaximumProgramVariables ||
        !same_raw_channel(prior.lane[3], source_word))
      return false;
    const std::uint32_t bit = 1u << variable;
    if ((bound & bit) != 0u && binding[variable] != source_word) return false;
    binding[variable] = source_word;
    bound |= bit;
    if (variables < variable + 1u) variables = variable + 1u;
  }
  for (std::uint32_t index = witness.reserved[1]; index < source.lane[2];
       ++index) {
    std::uint32_t deltas = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& delta = state->records[slot];
      if (delta.matter_q8 == 0u ||
          delta.lane[0] != kFormRevisionParticipationDelta ||
          delta.lane[1] != witness.lane[1] || delta.lane[2] != index)
        continue;
      std::uint32_t word = 0u;
      std::uint32_t meta = 0u;
      if (++deltas != 1u ||
          !program_term_at(state, source.lane[1], index, &word, &meta) ||
          delta.lane[3] != kInvalid || delta.lane[4] != kInvalid ||
          delta.lane[5] != word || delta.lane[6] != 0u ||
          delta.lane[7] != 2u || delta.reserved[0] != p4_owner ||
          delta.reserved[1] != trace_owner)
        return false;
    }
    if (deltas != 1u) return false;
  }
  if (mismatch != witness.lane[6] || variables == 0u) return false;
  *source_slot_out = source_slot;
  *variables_out = variables;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t
allocate_revision_regrowth_record(ResidentRewriteState* state) {
  if (state == nullptr) return kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormEmpty)
      continue;
    bool withdrawn_locus = false;
    for (std::uint32_t entry = 0u; entry < state->lesion.count; ++entry)
      withdrawn_locus |= state->lesion.original_slot[entry] == slot;
    if (withdrawn_locus) continue;
    candidate = Record{};
    candidate.matter_q8 = kRecordMatterQ8;
    return slot;
  }
  return kInvalid;
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t
available_revision_regrowth_records(const ResidentRewriteState* state) {
  if (state == nullptr) return 0u;
  std::uint32_t available = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormEmpty)
      continue;
    bool withdrawn_locus = false;
    for (std::uint32_t entry = 0u; entry < state->lesion.count; ++entry)
      withdrawn_locus |= state->lesion.original_slot[entry] == slot;
    available += !withdrawn_locus;
  }
  return available;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
reacquire_revision_reader_source(ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  std::uint32_t source_slot = kInvalid;
  std::uint32_t reader_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!pure_external_exact_program_authoritative(state, slot) ||
        !ticketed_revision_source_authoritative(state, slot))
      continue;
    const std::uint32_t candidate =
        unique_revision_reader_reacquisition_candidate(state, slot);
    if (candidate == kInvalid) continue;
    if (source_slot != kInvalid) return false;
    source_slot = slot;
    reader_slot = candidate;
  }
  if (source_slot == kInvalid || reader_slot == kInvalid) return false;
  const Record& source = state->records[source_slot];
  const std::uint32_t owner = state->records[reader_slot].lane[1];
  std::uint32_t p4_owner = kInvalid;
  std::uint32_t trace_owner = kInvalid;
  if (!revision_source_return_lineage(state, source_slot, &p4_owner,
                                      &trace_owner))
    return false;
  std::uint32_t delta_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormRevisionParticipationDelta ||
        delta.lane[1] != owner)
      continue;
    std::uint32_t word = 0u;
    std::uint32_t meta = 0u;
    if (delta.lane[2] >= source.lane[2] ||
        !revision_participation_reader_term_at(
            state, owner, delta.lane[2], &word, &meta) ||
        delta.lane[5] != word || delta.lane[6] != meta ||
        (delta.lane[7] != 1u && delta.lane[7] != 2u))
      return false;
    ++delta_count;
  }
  if (delta_count == 0u) return false;
  std::uint32_t witness_slot = kInvalid;
  std::uint32_t source_use_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[1] != owner) continue;
    if (record.lane[0] == kFormRevisionTransferWitness) {
      if (witness_slot != kInvalid) return false;
      witness_slot = slot;
    } else if (record.lane[0] == kFormRevisionTransferSourceUse) {
      if (source_use_slot != kInvalid) return false;
      source_use_slot = slot;
    }
  }
  if (witness_slot == kInvalid || source_use_slot == kInvalid) return false;
  Record& witness = state->records[witness_slot];
  Record& source_use = state->records[source_use_slot];
  witness.lane[3] = source.lane[1];
  witness.lane[5] = source.lane[5];
  ++witness.revision;
  source_use.lane[2] = source.lane[1];
  source_use.lane[3] = source.lane[5];
  ++source_use.revision;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormRevisionParticipationDelta ||
        delta.lane[1] != owner)
      continue;
    delta.reserved[0] = p4_owner;
    delta.reserved[1] = trace_owner;
    ++delta.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  return resident_revision_participation_reader_authoritative(state,
                                                               reader_slot);
}

#include "bcc32_resident_revision_transfer_regrowth.inl"

// GitHub #1060 (duplicate #1162 closed onto this ticket): the iterative,
// capacity-bounded replacement for the revision-transfer authority mutual
// recursion cycle (nvlink EIATTR_MIN_STACK_SIZE=0xffffffff / cuobjdump
// STACK:UNKNOWN for resident_rewrite_epoch_kernel and resident_epoch_post_
// return_kernel) lives in its own file so this one stays under the header
// line-budget ratchet; see the comment block at the top of that file for
// the full cycle map and design rationale.
#include "bcc32_resident_revision_lineage_bounded.inl"
