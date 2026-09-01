enum class ResidentEmissionPermitKind : std::uint32_t {
  kNone = 0u,
  kParticipationClosure = 1u,
};

// Physical overlay on a yielded trajectory.  The generic relation reader
// tolerates this legacy observer marker without importing inquiry semantics
// into its applicability or authority law.
inline constexpr std::uint32_t kTrajectoryOpenInquiry = 1u << 5u;

// Transaction-local permission for a composite resident authority to enter
// the one RWR0 writer. It names no content Record or answer locus. Digests are
// locators checked alongside a fresh structural/provenance rederivation; they
// can never authorize publication by themselves.
struct ResidentEmissionPermit {
  ResidentEmissionPermitKind kind = ResidentEmissionPermitKind::kNone;
  std::uint32_t owner = kInvalid;
  std::uint32_t participant_records = 0u;
  std::uint32_t external_leaves = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t source_contributions = 0u;
  // External source authority and live query/context integrity are separate
  // receipts. The latter can contain generated resident context but can never
  // authorize teaching or publication by itself.
  std::uint32_t source_external_leaves = 0u;
  std::uint32_t query_context_leaves = 0u;
  std::uint64_t topology_digest = 0u;
  std::uint64_t revision_digest = 0u;
  std::uint64_t provenance_digest = 0u;
  std::uint64_t query_context_digest = 0u;
};

struct ProgramCandidateConsensus {
  std::uint32_t word = 0u;
  std::uint32_t diagnostic_locus = kInvalid;
  std::uint32_t extent = 0u;
  // Equal-extent span candidates may use independently accrued support to
  // select a word, but only while no fixed or VersionSpace witness competes
  // at that extent.  These are transient collector facts, not resident state.
  std::uint32_t selected_support = 0u;
  bool have_candidate = false;
  bool conflict = false;
  bool selected_from_span = false;
  bool selected_from_causal_product = false;
  bool saw_nonspan_at_selected_extent = false;
  bool support_resolved_distinct_span = false;
  bool span_contributed = false;
  bool span_saw_program = false;
  bool span_saw_unbound = false;
  bool span_saw_ambiguous = false;
  bool version_space_saw_program = false;
  // A pending-means candidate is still an ordinary Program candidate. These
  // fields bind its exact resident causal provenance so the pending organ can
  // rederive it immediately before committing through the common emitter.
  bool selected_from_pending_means = false;
  std::uint32_t pending_header = kInvalid;
  std::uint32_t pending_header_revision = 0u;
  std::uint32_t pending_producer = kInvalid;
  std::uint32_t pending_producer_revision = 0u;
  std::uint32_t pending_target = 0u;
  std::uint32_t pending_action_revision = 0u;
  std::uint64_t pending_route_cost = 0u;
  std::uint32_t pending_route_depth = 0u;
  // Source-only relation candidate from the causal-constraint participation
  // ecology. These fields bind the exact const rederivation immediately before
  // the common emitter; they do not create a second publication path.
  bool selected_from_constraint_relation = false;
  bool constraint_relation_ambiguous = false;
  std::uint32_t constraint_relation_trajectory_owner = kInvalid;
  std::uint32_t constraint_relation_trajectory_revision = 0u;
  std::uint32_t constraint_relation = 0u;
  std::uint32_t constraint_relation_positive_sources = 0u;
  std::uint32_t constraint_relation_source_revision = 0u;
  std::uint32_t constraint_relation_probe_steps = 0u;
  std::uint32_t constraint_relation_participating_records = 0u;
  std::uint32_t constraint_relation_independent_sources = 0u;
  std::uint32_t constraint_relation_source_contributions = 0u;
  std::uint32_t constraint_relation_max_source_contribution = 0u;
  std::uint32_t constraint_relation_contribution_concentration_q16 = 0u;
  std::uint32_t constraint_relation_singleton_supported_steps = 0u;
  std::uint32_t constraint_relation_minimum_probe_support = 0u;
  std::uint64_t constraint_relation_component_digest = 0u;
  std::uint64_t constraint_relation_component_revision_digest = 0u;
  std::uint64_t constraint_relation_source_provenance_digest = 0u;
  std::uint32_t constraint_relation_source_external_leaves = 0u;
  std::uint64_t constraint_relation_query_context_digest = 0u;
  std::uint32_t constraint_relation_query_context_leaves = 0u;
  // Legacy aggregate receipt retained for existing observation/reafference
  // consumers. It is never used as the source-authority permit.
  std::uint64_t constraint_relation_external_provenance_digest = 0u;
  std::uint32_t constraint_relation_external_leaves = 0u;
  ResidentEmissionPermit emission_permit{};
  bool span_cursor_valid = false;
  std::uint32_t span_cursor_program = kInvalid;
  std::uint32_t span_cursor_start = 0u;
  std::uint32_t span_cursor_term = kInvalid;
  std::uint32_t span_cursor_offset = 0u;
  std::uint32_t span_cursor_lengths[kMaximumProgramVariables]{};
};

BCC32_REWRITE_HD inline bool emit_program_candidate_word(
    ResidentRewriteState* state, const ProgramCandidateConsensus& consensus);

BCC32_REWRITE_HD inline std::uint32_t observer_locus_for_resident_closure(
    const ResidentRewriteState* state, std::uint32_t trajectory_slot) {
  if (state == nullptr) return kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (slot == trajectory_slot) continue;
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] == kFormEmpty ||
        record.lane[0] == kFormMotor ||
        record.lane[0] == kFormConstraintParticipation ||
        record.lane[0] == kFormTrajectory ||
        record.lane[0] == kFormTrajectoryTerm ||
        record.lane[0] == kFormTrajectoryProvenance)
      continue;
    return slot;
  }
  return trajectory_slot;
}

BCC32_REWRITE_HD inline void merge_program_candidate(
    ProgramCandidateConsensus* consensus, std::uint32_t word,
    std::uint32_t diagnostic_locus, bool from_span,
    std::uint32_t extent, std::uint32_t support = 0u,
    bool from_causal_product = false);

BCC32_REWRITE_HD inline bool version_space_prefix_matches(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory) {
  if (program.lane[2] == 0u || trajectory.lane[2] + 1u != program.lane[2])
    return false;
  for (std::uint32_t index = 0u; index + 1u < program.lane[2]; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], index, &observed) ||
        !version_space_effective_program_term_at(state, program, index,
                                                 &expected, &meta))
      return false;
    if (observed != expected) return false;
  }
  return true;
}

BCC32_REWRITE_HD inline void collect_version_space_program_candidates(
    const ResidentRewriteState* state, const Record& trajectory,
    ProgramCandidateConsensus* consensus) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (program.lane[0] != kFormProgram ||
        (program.lane[7] & kProgramFlagVersionSpace) == 0u ||
        !resident_program_authoritative(state, slot) ||
        !version_space_prefix_matches(state, program, trajectory))
      continue;
    consensus->version_space_saw_program = true;
    merge_program_candidate(consensus, program.lane[6], slot, false,
                            trajectory.lane[2]);
  }
}

BCC32_REWRITE_HD inline void merge_program_candidate(
    ProgramCandidateConsensus* consensus, std::uint32_t word,
    std::uint32_t diagnostic_locus, bool from_span,
    std::uint32_t extent, std::uint32_t support, bool from_causal_product) {
  if (!consensus->have_candidate) {
    consensus->word = word;
    consensus->extent = extent;
    consensus->selected_support = from_span ? support : 0u;
    consensus->have_candidate = true;
    consensus->selected_from_span = from_span;
    consensus->selected_from_causal_product = from_causal_product;
    consensus->saw_nonspan_at_selected_extent = !from_span;
    consensus->support_resolved_distinct_span = false;
    consensus->span_contributed = from_span;
    // This is receipt provenance only and is never used to rank a word.
    consensus->diagnostic_locus = diagnostic_locus;
  } else if (extent > consensus->extent) {
    consensus->word = word;
    consensus->extent = extent;
    consensus->selected_support = from_span ? support : 0u;
    consensus->conflict = false;
    consensus->selected_from_span = from_span;
    consensus->selected_from_causal_product = from_causal_product;
    consensus->saw_nonspan_at_selected_extent = !from_span;
    consensus->support_resolved_distinct_span = false;
    consensus->diagnostic_locus = diagnostic_locus;
    consensus->span_contributed = from_span;
    consensus->span_cursor_valid = false;
  } else if (extent == consensus->extent) {
    if (consensus->saw_nonspan_at_selected_extent ||
        !consensus->selected_from_span || !from_span) {
      // A fixed Program or VersionSpace witness keeps the historical
      // conservative semantics.  In particular, it cannot ratify the word
      // selected earlier from a lower-supported distinct span witness.
      if (!from_span) consensus->saw_nonspan_at_selected_extent = true;
      if (consensus->word != word ||
          consensus->support_resolved_distinct_span) {
        consensus->conflict = true;
      } else if (!from_span && !consensus->selected_from_span &&
                 from_causal_product &&
                 !consensus->selected_from_causal_product) {
        // A causal-germline product is the more specific resident lineage
        // when it agrees with an older generic factor. Keep the word
        // consensus, but route the emission through the product locus so the
        // executable receipt names the exact factor-intersection lineage.
        consensus->diagnostic_locus = diagnostic_locus;
        consensus->selected_from_causal_product = true;
      } else if (diagnostic_locus < consensus->diagnostic_locus) {
        // Agreeing conservative witnesses retain stable provenance.
        consensus->diagnostic_locus = diagnostic_locus;
      }
    } else if (consensus->word == word) {
      if (support > consensus->selected_support) {
        // A stronger agreeing physical SpanProgram owns both the diagnostic
        // locus and any cursor installed by its collector.
        if (consensus->conflict)
          consensus->support_resolved_distinct_span = true;
        consensus->selected_support = support;
        consensus->diagnostic_locus = diagnostic_locus;
        consensus->conflict = false;
      } else if (support == consensus->selected_support &&
                 diagnostic_locus < consensus->diagnostic_locus) {
        // Equal-support agreement is stable across record traversal order.
        consensus->diagnostic_locus = diagnostic_locus;
      }
    } else if (support > consensus->selected_support) {
      // All candidates at this extent are spans, so a unique greater
      // independent-support witness lawfully replaces a different word.
      consensus->word = word;
      consensus->selected_support = support;
      consensus->diagnostic_locus = diagnostic_locus;
      consensus->support_resolved_distinct_span = true;
      consensus->conflict = false;
    } else if (support == consensus->selected_support) {
      // Different words tied at the maximum support still abstain.
      consensus->conflict = true;
    } else {
      // Keep the stronger selected word.  Remember the distinct lower
      // witness so a later non-span agreement cannot hide this alternative.
      consensus->support_resolved_distinct_span = true;
    }
  }
  if (from_span) consensus->span_contributed = true;
}

BCC32_REWRITE_HD inline bool collect_span_program_candidate_at(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t program_slot,
    std::uint32_t start, ProgramCandidateConsensus* consensus,
    bool* saw_program, bool* saw_unbound, bool* saw_ambiguous) {
  std::uint32_t bindings[kMaximumProgramVariables]
                        [kMaximumVariableSpanEvents]{};
  std::uint32_t lengths[kMaximumProgramVariables]{};
  std::uint32_t next_term = kInvalid;
  std::uint32_t next_offset = 0u;
  bool unbound = false;
  bool complete = false;
  bool ambiguous = false;
  const std::uint32_t match = span_match_prefix_at(
      state, program, trajectory, start, &next_term, &next_offset,
      &unbound, &complete, &ambiguous, bindings, lengths);
  if (match == kSpanMatchAmbiguous || ambiguous) {
    *saw_program = true;
    *saw_ambiguous = true;
    return true;
  }
  if (match != kSpanMatchPrefix) return false;
  if (complete || next_term == kInvalid) return true;
  *saw_program = true;
  if (unbound) {
    *saw_unbound = true;
    return true;
  }
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_left = 0u;
  std::uint32_t ignored_right = 0u;
  if (!span_program_term_at(state, program.lane[1], next_term, &kind,
                            &value, &ignored_left, &ignored_right))
    return true;
  std::uint32_t word = 0u;
  if (kind == kSpanTermLiteral) {
    word = value;
  } else if (value < kMaximumProgramVariables && lengths[value] != 0u &&
             next_offset < lengths[value]) {
    word = bindings[value][next_offset];
  } else {
    *saw_unbound = true;
    return true;
  }
  merge_program_candidate(consensus, word, program_slot, true,
                          trajectory.lane[2] - start, program.lane[3]);
  if (!consensus->conflict &&
      consensus->diagnostic_locus == program_slot) {
    consensus->span_cursor_valid = true;
    consensus->span_cursor_program = program_slot;
    consensus->span_cursor_start = start;
    consensus->span_cursor_term = next_term;
    consensus->span_cursor_offset = next_offset;
    for (std::uint32_t variable = 0u;
         variable < kMaximumProgramVariables; ++variable)
      consensus->span_cursor_lengths[variable] = lengths[variable];
  }
  return true;
}

BCC32_REWRITE_HD inline void collect_resident_span_program_candidates(
    const ResidentRewriteState* state, const Record& trajectory,
    ProgramCandidateConsensus* consensus) {
  if (state == nullptr || consensus == nullptr) return;
  bool saw_program = false;
  bool saw_unbound = false;
  bool saw_ambiguous = false;

  // Continue a program rooted in the external trajectory before considering
  // downstream programs rooted in generated matter. This avoids rescanning
  // every generated suffix while the first causal hop is still incomplete.
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& program = state->records[i];
    if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
        !resident_program_authoritative(state, i) ||
        program.lane[6] == trajectory.lane[1])
      continue;
    (void)collect_span_program_candidate_at(
        state, program, trajectory, i, 0u, consensus, &saw_program,
        &saw_unbound, &saw_ambiguous);
  }
  if (!consensus->have_candidate && !consensus->conflict && !saw_unbound &&
      !saw_ambiguous &&
      (trajectory.lane[7] & kTrajectoryHasGenerated) != 0u &&
      trajectory.lane[5] != kInvalid && trajectory.lane[5] < trajectory.lane[2]) {
    for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
      const Record& program = state->records[i];
      if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
          !resident_program_authoritative(state, i) ||
          program.lane[6] == trajectory.lane[1])
        continue;
      for (std::uint32_t start = trajectory.lane[5];
           start < trajectory.lane[2]; ++start) {
        if (start == 0u) continue;
        if (collect_span_program_candidate_at(
                state, program, trajectory, i, start, consensus,
                &saw_program, &saw_unbound, &saw_ambiguous))
          break;
      }
    }
  }
  consensus->span_saw_program |= saw_program;
  consensus->span_saw_unbound |= saw_unbound;
  consensus->span_saw_ambiguous |= saw_ambiguous;
}

BCC32_REWRITE_HD inline bool support_existing_programs(
    ResidentRewriteState* state, std::uint32_t trajectory_slot,
    bool include_fixed, bool include_span) {
  if (state == nullptr || trajectory_slot == kInvalid) return false;
  Record& trajectory = state->records[trajectory_slot];
  if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory ||
      trajectory.lane[7] != 0u)
    return false;
  ProgramSupportConsensus support{};
  collect_program_support_candidates(state, trajectory, &support,
                                     include_fixed, include_span);
  if (support.saw_ambiguous || support.conflict) {
    if (support.saw_ambiguous) ++state->span_ambiguous_abstentions;
    if (support.conflict) {
      ++state->program_conflict_abstentions;
      if (support.saw_span) ++state->span_conflict_abstentions;
    }
    refresh_receipt(state);
    return support.have_candidate || support.saw_ambiguous;
  }
  if (!support.have_candidate) return false;
  support_program_candidates(state, trajectory, support, include_fixed,
                             include_span);
  if (state->causal_germline_reflection_source != trajectory.lane[1])
    clear_trajectory(state, trajectory_slot);
  refresh_receipt(state);
  return true;
}

BCC32_REWRITE_HD inline bool support_existing_program(
    ResidentRewriteState* state, std::uint32_t trajectory_slot) {
  return support_existing_programs(state, trajectory_slot, true, false);
}

BCC32_REWRITE_HD inline bool support_existing_span_program(
    ResidentRewriteState* state, std::uint32_t trajectory_slot) {
  return support_existing_programs(state, trajectory_slot, false, true);
}

// Defined in bcc32_resident_causal_relation_reader.inl, included later in
// this same translation unit (line ~1269 below) -- forward-declared here
// since close_program_trajectory needs it before that include is reached.
BCC32_REWRITE_HD inline bool causal_relation_trajectory_ends_with_context(
    const ResidentRewriteState* state, const Record& trajectory);

// A mature causal Constructor is a stronger resident interpretation than a
// generic factor that happens to match the same raw suffix.  Preflight that
// interpretation while the source is still an external Trajectory so a page-
// spanning contact can be converted once and applied through the exact
// provenance path below.  This is only structural matching: no host answer,
// semantic label, or expected output is consulted.
BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_constructor_matches_trajectory(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t* constructor_slot, bool* ambiguous) {
  if (constructor_slot != nullptr) *constructor_slot = kInvalid;
  if (ambiguous != nullptr) *ambiguous = false;
  if (state == nullptr || constructor_slot == nullptr ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] == 0u)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& constructor = state->records[slot];
    if (constructor.matter_q8 == 0u ||
        constructor.lane[0] != kFormCausalConstructor)
      continue;
    if (!causal_constructor_authoritative(state, slot)) continue;
    bool matched = false;
    if (constructor.lane[6] == kFormProgram) {
      if (constructor.lane[2] != trajectory.lane[2]) continue;
      matched = true;
      for (std::uint32_t index = 0u; index < constructor.lane[2];
           ++index) {
        const std::uint32_t term_slot = causal_owned_ordinal(
            state, kFormCausalConstructorTerm, constructor.lane[1], index);
        std::uint32_t word = 0u;
        if (term_slot == kInvalid ||
            !causal_sequence_word_at(state, trajectory, index, &word) ||
            (word & kRawChannelMask) !=
                state->records[term_slot].lane[4]) {
          matched = false;
          break;
        }
      }
    } else if (constructor.lane[6] == kFormSpanProgram) {
      std::uint32_t lengths[kMaximumProgramVariables]{};
      bool local_ambiguous = false;
      matched = span_constructor_partition_sequence_structural(
          state, slot, trajectory, lengths, &local_ambiguous);
      if (local_ambiguous) {
        if (ambiguous != nullptr) *ambiguous = true;
        return false;
      }
    }
    if (!matched) continue;
    if (*constructor_slot != kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      *constructor_slot = kInvalid;
      return false;
    }
    *constructor_slot = slot;
  }
  return *constructor_slot != kInvalid;
}

// The production END worker performs bounded relation assimilation before the
// ordinary close phases. Preserve the exact external source long enough for
// the same counterevidence/causal-Constructor decision used by the direct
// close path; generic support remains owned by close_program_trajectory.
BCC32_CAUSAL_GERMLINE_DISPATCH bool
settle_external_causal_path_before_close(ResidentRewriteState* state) {
  if (state == nullptr) return false;
  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid) return false;
  const Record& current = state->records[current_slot];
  if (current.lane[0] != kFormTrajectory || current.lane[3] != 0u ||
      current.lane[4] != 0u || current.lane[5] != kInvalid ||
      current.lane[7] != 0u)
    return false;

  // Preflight existing causal products before the Constructor matcher. A
  // span Constructor deliberately matches reusable topology and channels;
  // one changed action/output literal is therefore still structurally
  // Constructor-compatible, but must revise the already-grounded product
  // rather than reacquire it as a fresh exact source. Generic
  // version-space/factor counterevidence remains deferred until after an
  // exact Constructor match below, so a new page-spanning witness is not
  // consumed by an older generic factor.
  GroundedCounterevidencePlan causal_counterevidence{};
  const GroundedCounterevidenceStatus causal_counter_status =
      preflight_grounded_counterevidence(
          state, current_slot, &causal_counterevidence, true);
  if (causal_counter_status == GroundedCounterevidenceStatus::kReady &&
      causal_counterevidence.constructor_slot != kInvalid) {
    if (!convert_pure_external_exact_trajectory(state, current_slot)) {
      state->records[current_slot].lane[3] = 1u;
      state->records[current_slot].lane[4] = 0u;
      ++state->records[current_slot].revision;
      return true;
    }
    if (!commit_grounded_counterevidence(
            state, current_slot, causal_counterevidence))
      state->fault = 0xc017u;
    refresh_receipt(state);
    return true;
  }

  // An exact, authoritative causal Constructor is more specific than a
  // generic version-space/product mismatch. Resolve that lineage first;
  // otherwise a one-literal generic counterevidence reader can consume a
  // valid novel witness before the factor-intersection Constructor gets to
  // see it.
  std::uint32_t constructor_slot = kInvalid;
  bool ambiguous = false;
  if (causal_constructor_matches_trajectory(
          state, current, &constructor_slot, &ambiguous)) {
    if (convert_pure_external_exact_trajectory(state, current_slot)) {
#if defined(__CUDA_ARCH__)
      state->causal_germline_application_program = current_slot;
#else
      (void)apply_causal_germline_to_exact_program(state, current_slot);
#endif
      return true;
    }
  }
  if (ambiguous) {
    ++state->causal_germline_conflict_abstentions;
    state->records[current_slot].lane[3] = 1u;
    state->records[current_slot].lane[4] = 0u;
    ++state->records[current_slot].revision;
    return true;
  }

  GroundedCounterevidencePlan counterevidence{};
  const GroundedCounterevidenceStatus counter_status =
      preflight_grounded_counterevidence(
          state, current_slot, &counterevidence);
  if (counter_status != GroundedCounterevidenceStatus::kNotApplicable) {
    if (!convert_pure_external_exact_trajectory(state, current_slot)) {
      state->records[current_slot].lane[3] = 1u;
      state->records[current_slot].lane[4] = 0u;
      ++state->records[current_slot].revision;
      return true;
    }
    if (counter_status == GroundedCounterevidenceStatus::kReady &&
        !commit_grounded_counterevidence(
            state, current_slot, counterevidence))
      state->fault = 0xc017u;
    refresh_receipt(state);
    return true;
  }
  return false;
}

BCC32_REWRITE_HD inline void close_program_trajectory(
    ResidentRewriteState* state,
    bool preserve_unclaimed_prefix = false) {
  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid) return;
  Record& current = state->records[current_slot];
  if (current.lane[2] == 0u) {
    clear_trajectory(state, current_slot);
    return;
  }
  if ((current.lane[7] & kTrajectoryHasCarry) != 0u) {
    // A generated Carry is already zero-authority execution context. It must
    // survive a later physical END so a fresh external connective can reach
    // the resident relation ecology; treating it like ordinary external
    // carry here would return before the Generated|Carry promotion branch
    // below and silently destroy the bridge. It remains excluded from source
    // support/induction, so retaining it cannot manufacture evidence.
    if ((current.lane[7] & kTrajectoryHasGenerated) != 0u) {
      current.lane[3] = 0u;
      current.lane[4] = 0u;
      ++current.revision;
    } else if (preserve_unclaimed_prefix &&
        (current.lane[7] & ~kTrajectoryHasCarry) == 0u) {
      current.lane[3] = 0u;
      current.lane[4] = 0u;
      ++current.revision;
    } else {
      clear_trajectory(state, current_slot);
    }
    return;
  }
  // A trajectory whose last word is a genuine, fully-tagged distributed or
  // generated event survives END as zero-authority execution context
  // instead of being retired: the resident bridge a later raw connective
  // needs, without the host ever re-supplying it. This grants no teaching
  // or support/induction authority -- the promoted Carry|Generated
  // phenotype is excluded from support_existing_programs and from the
  // relation reader's source-authority accounting; only the query-time
  // context path may read it.
  if ((current.lane[7] & kTrajectoryHasGenerated) != 0u &&
      causal_relation_trajectory_ends_with_context(state, current)) {
    current.lane[7] |= kTrajectoryHasCarry;
    current.lane[3] = 0u;
    current.lane[4] = 0u;
    ++current.revision;
    return;
  }
  // Generated words may re-enter the continuing trajectory and drive already
  // resident programs, but a mixed external/generated trajectory cannot
  // support or induce authority from its own output.
  if ((current.lane[7] & kTrajectoryHasGenerated) != 0u) {
    clear_trajectory(state, current_slot);
    return;
  }
  // Pause is an execution/query boundary, not belated developmental evidence.
  // If a caller later sends End only to close a yielded prefix that produced no
  // continuation, retire that partial instead of allowing it to factor with a
  // subsequent lived episode.  Explicit End without a prior Pause remains the
  // one-shot compilation boundary below.
  if (current.lane[4] != 0u ||
      (current.lane[7] & kTrajectoryWasYielded) != 0u) {
    clear_trajectory(state, current_slot);
    return;
  }
  // Fresh external evidence gets first opportunity to oppose an executable
  // learned product or to apply a uniquely matching causal Constructor.
  if (settle_external_causal_path_before_close(state)) return;
  // Preserve every compatible ordinary pair in the shared-factor ecology
  // before the legacy one-Program inducer performs its existing selection.
  bool version_space_divergent = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    const bool retained_trajectory =
        candidate.matter_q8 != 0u && candidate.lane[0] == kFormTrajectory &&
        candidate.lane[3] != 0u && candidate.lane[7] == 0u;
    const bool resident_literal =
        candidate.matter_q8 != 0u && candidate.lane[0] == kFormProgram &&
        (candidate.lane[7] & kProgramFlagPureExternalExact) != 0u &&
        literal_observation_source(state, slot);
    if ((!retained_trajectory && !resident_literal) || slot == current_slot)
      continue;
    bool divergent_consequence = false;
    (void)induce_version_space_from_pair(state, slot, current_slot,
                                         resident_literal,
                                         &divergent_consequence);
    // The compact VersionSpace inducer deliberately returns false when the
    // antecedent exceeds its bounded factor capacity; its independent
    // divergence receipt still blocks fixed induction and preserves the exact
    // current episode as a second resident literal Program.
    version_space_divergent |= divergent_consequence;
  }
  ProgramSupportConsensus support{};
  collect_program_support_candidates(state, current, &support);
  if (support.saw_ambiguous) ++state->span_ambiguous_abstentions;
  // A shorter fixed relation may already match while the complete fresh
  // episode supplies an additional grounded discriminator. Do not spend the
  // longer episode merely supporting that subsequence; let it reach exact and
  // learned-Constructor processing. Equal-extent and every span case keep the
  // existing support/abstention law.
  const bool longer_context_aperture =
      support.have_candidate && !support.saw_span &&
      !support.saw_ambiguous && support.maximum_fixed_extent != 0u &&
      support.maximum_fixed_extent < current.lane[2];
  if ((support.conflict || support.saw_ambiguous) &&
      !longer_context_aperture) {
    if (support.conflict) {
      ++state->program_conflict_abstentions;
      if (support.saw_span) ++state->span_conflict_abstentions;
    }
    current.lane[3] = 1u;
    current.lane[4] = 0u;
    ++current.revision;
    return;
  }
  if (support.have_candidate && !support.conflict &&
      !longer_context_aperture) {
    support_program_candidates(state, current, support);
    if (state->causal_germline_reflection_source != current.lane[1])
      clear_trajectory(state, current_slot);
    return;
  }

  // Identical antecedents with different lived consequences are an explicit
  // resident version space. Do not collapse that evidence into a fixed Program
  // whose output is an unbound variable. Retain the new exact interpretation;
  // later independent owners can strengthen or specialize either alternative.
  if (version_space_divergent) {
    if (convert_pure_external_exact_trajectory(state, current_slot)) {
#if defined(__CUDA_ARCH__)
      state->causal_germline_application_program = current_slot;
#else
      (void)apply_causal_germline_to_exact_program(state, current_slot);
#endif
      return;
    }
    current.lane[3] = 1u;
    current.lane[4] = 0u;
    ++current.revision;
    return;
  }

  // Derive every fixed and variable-span induction identity before allowing
  // either inducer to mutate Records. Equal identities share one canonical
  // organization; divergent identities abstain instead of using slot order.
  std::uint32_t fixed_identity = kInvalid;
  std::uint32_t fixed_left = kInvalid;
  std::uint32_t fixed_right = kInvalid;
  std::uint32_t span_identity = kInvalid;
  std::uint32_t span_left = kInvalid;
  std::uint32_t span_right = kInvalid;
  bool induction_conflict = false;
  // The Genesis constructor is ordinary resident bootstrap matter. Lesioning
  // it freezes the old pair/span teacher without disabling exact external
  // memory or constructors the adult already learned from that teacher.
  const bool bootstrap_induction_available =
      find_form(state, kFormConstructor) != kInvalid;
  for (std::uint32_t i = 0u;
       bootstrap_induction_available && i < live_record_capacity(state); ++i) {
    const Record& candidate = state->records[i];
    const bool retained_trajectory =
        candidate.matter_q8 != 0u && candidate.lane[0] == kFormTrajectory &&
        candidate.lane[3] != 0u && candidate.lane[7] == 0u;
    const bool resident_literal =
        candidate.matter_q8 != 0u && candidate.lane[0] == kFormProgram &&
        (candidate.lane[7] & kProgramFlagPureExternalExact) != 0u &&
        literal_observation_source(state, i);
    if ((!retained_trajectory && !resident_literal) || i == current_slot)
      continue;
    std::uint32_t identity = kInvalid;
    if (induce_program(state, i, current_slot, &identity, false,
                       resident_literal)) {
      if (fixed_identity == kInvalid) {
        fixed_identity = identity;
        fixed_left = i;
        fixed_right = current_slot;
      } else if (fixed_identity != identity) {
        induction_conflict = true;
      }
    }
    identity = kInvalid;
    if (induce_span_program(state, i, current_slot, &identity, false,
                            resident_literal)) {
      if (span_identity == kInvalid) {
        span_identity = identity;
        span_left = i;
        span_right = current_slot;
      } else if (span_identity != identity) {
        induction_conflict = true;
      }
    }
  }
  if (fixed_identity != kInvalid && span_identity != kInvalid)
    induction_conflict = true;
  if (!induction_conflict && fixed_identity != kInvalid) {
    if (induce_program(
            state, fixed_left, fixed_right, nullptr, true,
            state->records[fixed_left].lane[0] == kFormProgram))
      return;
  }
  if (!induction_conflict && span_identity != kInvalid) {
    if (induce_span_program(
            state, span_left, span_right, nullptr, true,
            state->records[span_left].lane[0] == kFormProgram))
      return;
  }
  // Exact resident memory and VersionSpace alternatives are complementary
  // interpretations.  When no fixed or unequal-span transformation claims the
  // episode, retain the exact executable Program even if compact ambiguity
  // evidence was also admitted above.
  if (preserve_unclaimed_prefix && !induction_conflict &&
      fixed_identity == kInvalid && span_identity == kInvalid) {
    // A physical contact may end at a unique, fully bound prefix of mature
    // resident span matter.  Existing support, VersionSpace, and induction
    // paths above retain first authority.  Only the otherwise-unclaimed prefix
    // remains trajectory matter so the cross-contact layer can carry it;
    // RWR11 must not compile an incomplete relation into exact source memory.
    current.lane[3] = 1u;
    current.lane[4] = 0u;
    ++current.revision;
    return;
  }
  if (!induction_conflict && fixed_identity == kInvalid &&
      span_identity == kInvalid &&
      convert_pure_external_exact_trajectory(state, current_slot)) {
#if defined(__CUDA_ARCH__)
    state->causal_germline_application_program = current_slot;
#else
    (void)apply_causal_germline_to_exact_program(state, current_slot);
#endif
    return;
  }
  // A malformed or otherwise incomplete literal chain does not gain
  // authority. Seal the boundary as the existing non-authoritative retained
  // exemplar path so the next external episode cannot merge with it.
  current.lane[3] = 1u;
  current.lane[4] = 0u;
  ++current.revision;
}

BCC32_REWRITE_HD inline void yield_program_trajectory(
    ResidentRewriteState* state) {
  const std::uint32_t current = find_current_trajectory(state);
  if (current == kInvalid) return;
  state->records[current].lane[4] = 1u;
  state->records[current].lane[7] |= kTrajectoryWasYielded;
  ++state->records[current].revision;
}

BCC32_REWRITE_HD inline bool match_program_prefix_at(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t start_offset,
    std::uint32_t* next_word,
    bool* next_is_unbound) {
  if ((program.lane[7] & kProgramFlagVersionSpace) != 0u ||
      start_offset >= trajectory.lane[2])
    return false;
  const std::uint32_t count = trajectory.lane[2] - start_offset;
  if (count == 0u || count >= program.lane[2]) return false;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  for (std::uint32_t index = 0u; index < count; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], start_offset + index,
                            &observed) ||
        !program_term_at(state, program.lane[1], index, &expected, &meta))
      return false;
    if (meta == 0u) {
      if (observed != expected) return false;
      continue;
    }
    const std::uint32_t variable = meta - 1u;
    if (variable >= kMaximumProgramVariables ||
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
  std::uint32_t expected = 0u;
  std::uint32_t meta = 0u;
  if (!program_term_at(state, program.lane[1], count, &expected, &meta))
    return false;
  if (meta == 0u) {
    *next_word = expected;
    *next_is_unbound = false;
    return true;
  }
  const std::uint32_t variable = meta - 1u;
  if (variable >= kMaximumProgramVariables ||
      (bound & (1u << variable)) == 0u) {
    *next_is_unbound = true;
    return true;
  }
  *next_word = binding[variable];
  *next_is_unbound = false;
  return true;
}

BCC32_REWRITE_HD inline bool match_program_prefix(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t* next_word,
    bool* next_is_unbound) {
  return match_program_prefix_at(state, program, trajectory, 0u, next_word,
                                 next_is_unbound);
}

BCC32_REWRITE_HD inline bool collect_fixed_program_candidate_at(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t program_slot,
    std::uint32_t start, ProgramCandidateConsensus* consensus,
    bool* saw_unbound) {
  std::uint32_t word = 0u;
  bool unbound = false;
  if (!match_program_prefix_at(state, program, trajectory, start, &word,
                               &unbound))
    return false;
  if (unbound) {
    *saw_unbound = true;
    return true;
  }
  merge_program_candidate(consensus, word, program_slot, false,
                          trajectory.lane[2] - start, 0u,
                          (program.lane[7] &
                           kProgramFlagCausalGermlineProduct) != 0u);
  return true;
}

BCC32_REWRITE_HD inline void collect_fixed_program_candidates(
    const ResidentRewriteState* state, const Record& trajectory,
    ProgramCandidateConsensus* consensus, bool* saw_unbound) {
  if (state == nullptr || consensus == nullptr || saw_unbound == nullptr)
    return;
  // A fresh physical reply may select one live VersionSpace consequence for
  // this exact suspended trajectory. Only that bounded selection subordinates
  // older fixed Programs for the next step. Ordinary unresolved VersionSpace
  // trajectories still collect every authority and therefore abstain on equal
  // divergent alternatives.
  if (consensus->version_space_saw_program &&
      (trajectory.lane[7] & kTrajectoryVersionSpaceSelected) != 0u)
    return;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& program = state->records[i];
    if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        (program.lane[7] & kProgramFlagVersionSpace) != 0u ||
        !resident_program_authoritative(state, i) ||
        program.lane[6] == trajectory.lane[1])
      continue;
    (void)collect_fixed_program_candidate_at(
        state, program, trajectory, i, 0u, consensus, saw_unbound);
  }
  if (consensus->have_candidate || consensus->conflict || *saw_unbound ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u ||
      trajectory.lane[5] == kInvalid || trajectory.lane[5] >= trajectory.lane[2])
    return;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& program = state->records[i];
    if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        (program.lane[7] & kProgramFlagVersionSpace) != 0u ||
        !resident_program_authoritative(state, i) ||
        program.lane[6] == trajectory.lane[1])
      continue;
    for (std::uint32_t start = trajectory.lane[5];
         start < trajectory.lane[2]; ++start) {
      if (start != 0u && collect_fixed_program_candidate_at(
                             state, program, trajectory, i, start, consensus,
                             saw_unbound))
        break;
    }
  }
}

// Once a generated word completes a mature program, settle every agreeing
// witness for this trajectory owner. A variable-span program remains eligible
// while its ordered resident span is still being emitted.
BCC32_REWRITE_HD inline void settle_program_candidates(
    ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr) return;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    Record& program = state->records[i];
    if (!resident_program_authoritative(state, i) ||
        program.lane[6] == trajectory.lane[1])
      continue;
    bool agrees = false;
    if (program.lane[0] == kFormProgram) {
      if ((program.lane[7] & kProgramFlagVersionSpace) != 0u) continue;
      agrees = full_program_match(state, program, trajectory);
    } else if (program.lane[0] == kFormSpanProgram) {
      bool ambiguous = false;
      agrees = full_span_program_match(state, program, trajectory, &ambiguous) &&
               !ambiguous;
    }
    if (agrees) program.lane[6] = trajectory.lane[1];
  }
}

BCC32_REWRITE_HD inline std::uint32_t find_span_execution_cursor(
    const ResidentRewriteState* state, std::uint32_t trajectory_owner) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == kFormSpanExecutionCursor &&
        record.lane[1] == trajectory_owner) {
      if (found != kInvalid) return kInvalid;
      found = slot;
    }
  }
  return found;
}

BCC32_REWRITE_HD inline bool span_execution_cursor_collision(
    const ResidentRewriteState* state, std::uint32_t trajectory_owner) {
  if (state == nullptr) return false;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == kFormSpanExecutionCursor &&
        record.lane[1] == trajectory_owner && ++count > 1u)
      return true;
  }
  return false;
}

// Cursor execution never retains a physical provider slot.  Recompute the
// resident SpanProgram identity from its exact term population, then require
// one owner+revision+identity provider.  Duplicate headers or terms are an
// authority collision and therefore fail closed rather than inheriting array
// traversal order as a program selector.
BCC32_REWRITE_HD inline bool span_program_structural_identity(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t* identity) {
  if (state == nullptr || identity == nullptr ||
      program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
      program.lane[2] == 0u ||
      program.lane[2] > kMaximumSpanProgramTerms ||
      program.lane[4] == 0u ||
      program.lane[4] > kMaximumProgramVariables)
    return false;
  std::uint32_t term_records = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormSpanProgramTerm ||
        term.lane[1] != program.lane[1])
      continue;
    if (term.lane[2] >= program.lane[2]) return false;
    ++term_records;
  }
  if (term_records != program.lane[2]) return false;

  std::uint32_t digest = rewrite_mix(
      kFormSpanProgram, program.lane[2], program.lane[4]);
  for (std::uint32_t ordinal = 0u; ordinal < program.lane[2]; ++ordinal) {
    const Record* selected = nullptr;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      if (term.matter_q8 == 0u || term.lane[0] != kFormSpanProgramTerm ||
          term.lane[1] != program.lane[1] || term.lane[2] != ordinal)
        continue;
      if (selected != nullptr) return false;
      selected = &term;
    }
    if (selected == nullptr ||
        (selected->lane[3] != kSpanTermLiteral &&
         selected->lane[3] != kSpanTermVariable) ||
        (selected->lane[3] == kSpanTermVariable &&
         selected->lane[4] >= program.lane[4]) ||
        selected->lane[6] != 0u || selected->lane[7] != 0u ||
        selected->reserved[0] != 0u || selected->reserved[1] != 0u)
      return false;
    digest = rewrite_mix(digest, selected->lane[3],
                         selected->lane[4] ^ selected->lane[5]);
  }
  *identity = digest;
  return true;
}

BCC32_REWRITE_HD inline std::uint32_t resolve_span_cursor_provider(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t revision, std::uint32_t structural_identity) {
  if (state == nullptr || owner == 0u || owner == kInvalid ||
      revision == 0u)
    return kInvalid;
  std::uint32_t provider = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormSpanProgram ||
        candidate.lane[1] != owner || candidate.lane[5] != structural_identity ||
        static_cast<std::uint32_t>(candidate.revision) != revision)
      continue;
    std::uint32_t derived_identity = kInvalid;
    if (!span_program_structural_identity(state, candidate,
                                          &derived_identity) ||
        derived_identity != structural_identity ||
        !resident_program_authoritative(state, slot))
      continue;
    if (provider != kInvalid) return kInvalid;
    provider = slot;
  }
  return provider;
}

BCC32_REWRITE_HD inline std::uint32_t find_span_execution_binding(
    const ResidentRewriteState* state, std::uint32_t trajectory_owner,
    std::uint32_t program_owner, std::uint32_t variable) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormSpanExecutionBinding ||
        binding.lane[1] != trajectory_owner || binding.lane[2] != variable ||
        binding.lane[4] != program_owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline bool span_execution_binding_length(
    const ResidentRewriteState* state, std::uint32_t trajectory_owner,
    std::uint32_t program_owner, std::uint32_t variable,
    std::uint32_t program_revision, std::uint32_t program_identity,
    std::uint32_t* length) {
  if (length == nullptr) return false;
  const std::uint32_t slot = find_span_execution_binding(
      state, trajectory_owner, program_owner, variable);
  if (slot == kInvalid) return false;
  const Record& binding = state->records[slot];
  if (binding.lane[3] == 0u ||
      binding.lane[3] > kMaximumVariableSpanEvents ||
      binding.lane[5] != program_revision ||
      binding.lane[6] != program_identity ||
      binding.lane[7] != 0u || binding.reserved[0] != 0u ||
      binding.reserved[1] != 0u)
    return false;
  *length = binding.lane[3];
  return true;
}

BCC32_REWRITE_HD inline void clear_span_execution_bindings(
    ResidentRewriteState* state, std::uint32_t trajectory_owner) {
  if (state == nullptr) return;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& binding = state->records[slot];
    if (binding.matter_q8 != 0u &&
        binding.lane[0] == kFormSpanExecutionBinding &&
        binding.lane[1] == trajectory_owner)
      clear_record(&binding);
  }
}

BCC32_REWRITE_HD inline bool span_binding_start(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t trajectory_owner, std::uint32_t trajectory_start,
    std::uint32_t target_variable, std::uint32_t* binding_start) {
  if (state == nullptr || binding_start == nullptr) return false;
  std::uint32_t cursor = trajectory_start;
  for (std::uint32_t term_index = 0u; term_index < program.lane[2];
       ++term_index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t ignored_left = 0u;
    std::uint32_t ignored_right = 0u;
    if (!span_program_term_at(state, program.lane[1], term_index, &kind,
                              &value, &ignored_left, &ignored_right))
      return false;
    if (kind == kSpanTermLiteral) {
      ++cursor;
      continue;
    }
    if (kind != kSpanTermVariable || value >= kMaximumProgramVariables)
      return false;
    std::uint32_t length = 0u;
    if (!span_execution_binding_length(
            state, trajectory_owner, program.lane[1], value,
            static_cast<std::uint32_t>(program.revision), program.lane[5],
            &length))
      return false;
    if (value == target_variable) {
      *binding_start = cursor;
      return true;
    }
    cursor += length;
  }
  return false;
}

BCC32_REWRITE_HD inline bool next_span_cursor_position(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t term_index, std::uint32_t term_offset,
    const std::uint32_t lengths[kMaximumProgramVariables],
    std::uint32_t* next_term, std::uint32_t* next_offset) {
  if (state == nullptr || next_term == nullptr || next_offset == nullptr ||
      term_index >= program.lane[2])
    return false;
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_left = 0u;
  std::uint32_t ignored_right = 0u;
  if (!span_program_term_at(state, program.lane[1], term_index, &kind,
                            &value, &ignored_left, &ignored_right))
    return false;
  if (kind == kSpanTermLiteral) {
    *next_term = term_index + 1u;
    *next_offset = 0u;
    return true;
  }
  if (kind != kSpanTermVariable || value >= kMaximumProgramVariables ||
      lengths[value] == 0u || term_offset >= lengths[value])
    return false;
  if (term_offset + 1u < lengths[value]) {
    *next_term = term_index;
    *next_offset = term_offset + 1u;
  } else {
    *next_term = term_index + 1u;
    *next_offset = 0u;
  }
  return true;
}

BCC32_REWRITE_HD inline bool next_span_cursor_position_from_records(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t trajectory_owner, std::uint32_t term_index,
    std::uint32_t term_offset, std::uint32_t* next_term,
    std::uint32_t* next_offset) {
  if (state == nullptr || next_term == nullptr || next_offset == nullptr ||
      term_index >= program.lane[2])
    return false;
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_left = 0u;
  std::uint32_t ignored_right = 0u;
  if (!span_program_term_at(state, program.lane[1], term_index, &kind, &value,
                            &ignored_left, &ignored_right))
    return false;
  if (kind == kSpanTermLiteral) {
    *next_term = term_index + 1u;
    *next_offset = 0u;
    return true;
  }
  if (kind != kSpanTermVariable || value >= kMaximumProgramVariables)
    return false;
  std::uint32_t length = 0u;
  if (!span_execution_binding_length(
          state, trajectory_owner, program.lane[1], value,
          static_cast<std::uint32_t>(program.revision), program.lane[5],
          &length) ||
      term_offset >= length)
    return false;
  if (term_offset + 1u < length) {
    *next_term = term_index;
    *next_offset = term_offset + 1u;
  } else {
    *next_term = term_index + 1u;
    *next_offset = 0u;
  }
  return true;
}

BCC32_REWRITE_HD inline bool install_span_execution_cursor(
    ResidentRewriteState* state, const Record& trajectory,
    const ProgramCandidateConsensus& consensus) {
  if (state == nullptr || !consensus.span_cursor_valid ||
      consensus.span_cursor_program != consensus.diagnostic_locus ||
      consensus.span_cursor_program >= live_record_capacity(state))
    return false;
  const Record& program = state->records[consensus.span_cursor_program];
  if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
      program.lane[4] == 0u || program.lane[4] > kMaximumProgramVariables)
    return false;
  std::uint32_t program_identity = kInvalid;
  if (!span_program_structural_identity(state, program, &program_identity) ||
      program_identity != program.lane[5] ||
      resolve_span_cursor_provider(
          state, program.lane[1],
          static_cast<std::uint32_t>(program.revision), program_identity) !=
          consensus.span_cursor_program)
    return false;
  std::uint32_t next_term = kInvalid;
  std::uint32_t next_offset = 0u;
  if (!next_span_cursor_position(
          state, program, consensus.span_cursor_term,
          consensus.span_cursor_offset, consensus.span_cursor_lengths,
          &next_term, &next_offset) ||
      next_term >= program.lane[2])
    return false;

  const std::uint32_t trajectory_owner = trajectory.lane[1];
  const std::uint32_t program_owner = program.lane[1];
  const std::uint32_t program_revision =
      static_cast<std::uint32_t>(program.revision);
  std::uint32_t cursor_slot =
      find_span_execution_cursor(state, trajectory_owner);
  if (cursor_slot == kInvalid &&
      span_execution_cursor_collision(state, trajectory_owner))
    return false;
  if (cursor_slot != kInvalid &&
      (state->records[cursor_slot].lane[2] != program_owner ||
       state->records[cursor_slot].lane[7] != program_revision ||
       state->records[cursor_slot].reserved[0] != program_identity)) {
    clear_record(&state->records[cursor_slot]);
    clear_span_execution_bindings(state, trajectory_owner);
    cursor_slot = kInvalid;
  } else if (cursor_slot == kInvalid) {
    clear_span_execution_bindings(state, trajectory_owner);
  }

  // Remove bindings left by a prior variable population while preserving
  // valid bindings for this exact resident Program.
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& binding = state->records[slot];
    if (binding.matter_q8 != 0u &&
        binding.lane[0] == kFormSpanExecutionBinding &&
        binding.lane[1] == trajectory_owner &&
        (binding.lane[4] != program_owner ||
         binding.lane[5] != program_revision ||
         binding.lane[6] != program_identity ||
         binding.lane[2] >= program.lane[4]))
      clear_record(&binding);
  }

  std::uint32_t missing_bindings = 0u;
  for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable) {
    const std::uint32_t binding_slot = find_span_execution_binding(
        state, trajectory_owner, program_owner, variable);
    if (binding_slot == kInvalid) {
      std::uint32_t matches = 0u;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& binding = state->records[slot];
        if (binding.matter_q8 != 0u &&
            binding.lane[0] == kFormSpanExecutionBinding &&
            binding.lane[1] == trajectory_owner &&
            binding.lane[2] == variable &&
            binding.lane[4] == program_owner)
          ++matches;
      }
      if (matches > 1u) return false;
      ++missing_bindings;
    }
  }
  const std::uint32_t required = missing_bindings +
      (cursor_slot == kInvalid ? 1u : 0u);
  if (free_record_count(state) < required) {
    state->fault = 1u;
    return false;
  }
  if (cursor_slot == kInvalid) {
    cursor_slot = allocate_record(state);
    if (cursor_slot == kInvalid) return false;
  }

  for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable) {
    std::uint32_t binding_slot = find_span_execution_binding(
        state, trajectory_owner, program_owner, variable);
    if (binding_slot == kInvalid) {
      binding_slot = allocate_record(state);
      if (binding_slot == kInvalid) return false;
    }
    Record& binding = state->records[binding_slot];
    binding = Record{};
    binding.lane[0] = kFormSpanExecutionBinding;
    binding.lane[1] = trajectory_owner;
    binding.lane[2] = variable;
    binding.lane[3] = consensus.span_cursor_lengths[variable];
    binding.lane[4] = program_owner;
    binding.lane[5] = program_revision;
    binding.lane[6] = program_identity;
    if (binding.lane[3] == 0u ||
        binding.lane[3] > kMaximumVariableSpanEvents)
      return false;
    binding.revision = 1u;
    binding.matter_q8 = kRecordMatterQ8;
  }

  Record& cursor = state->records[cursor_slot];
  cursor = Record{};
  cursor.lane[0] = kFormSpanExecutionCursor;
  cursor.lane[1] = trajectory_owner;
  cursor.lane[2] = program_owner;
  cursor.lane[3] = consensus.span_cursor_start;
  cursor.lane[4] = next_term;
  cursor.lane[5] = next_offset;
  cursor.lane[6] = program.lane[4];
  cursor.lane[7] = program_revision;
  cursor.reserved[0] = program_identity;
  cursor.revision = 1u;
  cursor.matter_q8 = kRecordMatterQ8;
  ++state->revision;
  return true;
}

BCC32_REWRITE_HD inline bool advance_span_execution_cursor(
    ResidentRewriteState* state, std::uint32_t trajectory_slot) {
  if (state == nullptr || trajectory_slot == kInvalid) return false;
  Record& trajectory = state->records[trajectory_slot];
  const std::uint32_t cursor_slot =
      find_span_execution_cursor(state, trajectory.lane[1]);
  if (cursor_slot == kInvalid) return false;
  Record& cursor = state->records[cursor_slot];
  const std::uint32_t program_slot = resolve_span_cursor_provider(
      state, cursor.lane[2], cursor.lane[7], cursor.reserved[0]);
  if (program_slot == kInvalid || cursor.reserved[1] != 0u) {
    clear_record(&cursor);
    clear_span_execution_bindings(state, trajectory.lane[1]);
    return false;
  }
  const Record& program = state->records[program_slot];
  if (cursor.lane[6] != program.lane[4] ||
      cursor.lane[7] != static_cast<std::uint32_t>(program.revision) ||
      cursor.lane[4] >= program.lane[2]) {
    clear_record(&cursor);
    clear_span_execution_bindings(state, trajectory.lane[1]);
    return false;
  }
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored_left = 0u;
  std::uint32_t ignored_right = 0u;
  if (!span_program_term_at(state, program.lane[1], cursor.lane[4], &kind,
                            &value, &ignored_left, &ignored_right)) {
    clear_record(&cursor);
    return false;
  }
  std::uint32_t word = 0u;
  if (kind == kSpanTermLiteral) {
    word = value;
  } else if (kind == kSpanTermVariable && value < kMaximumProgramVariables) {
    std::uint32_t length = 0u;
    if (!span_execution_binding_length(
            state, trajectory.lane[1], program.lane[1], value,
            static_cast<std::uint32_t>(program.revision), program.lane[5],
            &length) ||
        cursor.lane[5] >= length) {
      clear_record(&cursor);
      clear_span_execution_bindings(state, trajectory.lane[1]);
      return false;
    }
    std::uint32_t binding_start = 0u;
    if (!span_binding_start(state, program, trajectory.lane[1],
                            cursor.lane[3], value, &binding_start) ||
        !trajectory_word_at(state, trajectory.lane[1],
                            binding_start + cursor.lane[5], &word)) {
      clear_record(&cursor);
      clear_span_execution_bindings(state, trajectory.lane[1]);
      return false;
    }
  } else {
    clear_record(&cursor);
    clear_span_execution_bindings(state, trajectory.lane[1]);
    return false;
  }
  std::uint32_t next_term = kInvalid;
  std::uint32_t next_offset = 0u;
  if (!next_span_cursor_position_from_records(
          state, program, trajectory.lane[1], cursor.lane[4], cursor.lane[5],
          &next_term, &next_offset)) {
    clear_record(&cursor);
    clear_span_execution_bindings(state, trajectory.lane[1]);
    return false;
  }
  ProgramCandidateConsensus emitted{};
  emitted.word = word;
  emitted.diagnostic_locus = program_slot;
  emitted.extent = trajectory.lane[2];
  emitted.have_candidate = true;
  emitted.selected_from_span = true;
  emitted.span_contributed = true;
  if (!emit_program_candidate_word(state, emitted)) return false;
  if (next_term >= program.lane[2]) {
    settle_program_candidates(state, trajectory);
    clear_record(&cursor);
    clear_span_execution_bindings(state, trajectory.lane[1]);
  } else {
    cursor.lane[4] = next_term;
    cursor.lane[5] = next_offset;
    ++cursor.revision;
  }
  refresh_receipt(state);
  return true;
}

// Raw SpanProgram composition participates in the same extent-first,
// conservative consensus as every other resident continuation.  These
// helpers depend on ProgramCandidateConsensus, its merge law, and the span
// cursor queries above; neither helper introduces a second output authority.
#define BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE 1
#include "bcc32_resident_raw_span_composition.inl"
#include "bcc32_resident_raw_span_consensus_binding.inl"
#undef BCC32_RAW_SPAN_INSIDE_REWRITE_NAMESPACE
#include "bcc32_resident_causal_relation_source_witness.cuh"
#define BCC32_TURN_WORLD_BINDING_INSIDE_REWRITE_NAMESPACE 1
#include "bcc32_resident_turn_world_consequence_binding.cuh"
#undef BCC32_TURN_WORLD_BINDING_INSIDE_REWRITE_NAMESPACE
#include "bcc32_resident_causal_relation_reader.inl"

// One common commit rail owns ordinary Program-generated public words.  The
// caller remains responsible for candidate-specific trajectory settlement and
// exact provenance stamping, but no Program consumer may write the public
// generated-word registers on a parallel path.
BCC32_REWRITE_HD inline bool emit_program_candidate_word(
    ResidentRewriteState* state, const ProgramCandidateConsensus& consensus) {
  const bool resident_closure =
      consensus.emission_permit.kind != ResidentEmissionPermitKind::kNone;
  const std::uint32_t trajectory_slot =
      state == nullptr ? kInvalid : find_current_trajectory(state);
  const std::uint32_t emitted_index =
      trajectory_slot == kInvalid ? kInvalid
                                  : state->records[trajectory_slot].lane[2];
  const bool permit_authoritative =
      resident_closure
          ? resident_closure_emission_permit_authoritative(state, consensus)
          : resident_program_authoritative(state, consensus.diagnostic_locus);
  if (state == nullptr || state->fault != 0u || !consensus.have_candidate ||
      consensus.conflict ||
      (!resident_closure &&
       (consensus.diagnostic_locus == kInvalid ||
        consensus.diagnostic_locus >= live_record_capacity(state))) ||
      (resident_closure &&
       (consensus.diagnostic_locus != kInvalid ||
        trajectory_slot == kInvalid || emitted_index == kInvalid ||
        consensus.emission_permit.owner !=
            state->records[trajectory_slot].lane[1])) ||
      state->program_generated_events == ~std::uint32_t{0} ||
      (consensus.selected_from_constraint_relation &&
       state->causal_relation_generated_events == ~std::uint32_t{0}) ||
      (consensus.span_contributed &&
       state->span_generated_events == ~std::uint32_t{0}) ||
      !permit_authoritative ||
      !append_trajectory_word(state, consensus.word, true))
    return false;
  // A later ordinary generated term supersedes any earlier distributed
  // closure anchor on this yielded trajectory.  Without this reset, a stale
  // resident relation term could authorize a returned consequence for a
  // newer ordinary producer merely because both share lane[5].
  if (!resident_closure && trajectory_slot != kInvalid &&
      state->causal_relation_trajectory_owner ==
          state->records[trajectory_slot].lane[1]) {
    state->causal_relation_trajectory_owner = kInvalid;
    state->causal_relation_generated_index = kInvalid;
    state->causal_relation_trajectory_revision = 0u;
  }
  state->generated_word = consensus.word;
  state->generated_word_valid = 1u;
  state->generated_locus = resident_closure ? kInvalid
                                            : consensus.diagnostic_locus;
  // A distributed closure has no single producer locus, but the last
  // resident action remains a valid observer-selected lesion target for the
  // subsequent focal-cut assay.  It is never consulted by the emission
  // permit or any relation authority check.
  state->active_locus = resident_closure
      ? observer_locus_for_resident_closure(state, trajectory_slot)
      : state->generated_locus;
  if (resident_closure) {
    const ResidentEmissionPermit& permit = consensus.emission_permit;
    state->generated_receipt_owner = permit.owner;
    state->generated_receipt_participant_records = permit.participant_records;
    state->generated_receipt_external_leaves = permit.external_leaves;
    state->generated_receipt_independent_sources = permit.independent_sources;
    state->generated_receipt_source_contributions = permit.source_contributions;
    state->generated_receipt_topology_digest = permit.topology_digest;
    state->generated_receipt_revision_digest = permit.revision_digest;
    state->generated_receipt_provenance_digest = permit.provenance_digest;
    state->generated_receipt_participation_digest = causal_relation_mix(
        causal_relation_mix(permit.topology_digest, permit.revision_digest),
        permit.provenance_digest);
    state->generated_receipt_epoch = state->revision;
    state->generated_receipt_valid = 1u;
    state->causal_relation_trajectory_owner = permit.owner;
    state->causal_relation_generated_index = emitted_index;
  }
  ++state->program_generated_events;
  if (consensus.span_contributed) ++state->span_generated_events;
  if (consensus.selected_from_constraint_relation) {
    ++state->causal_relation_generated_events;
    state->causal_relation_probe_steps =
        consensus.constraint_relation_probe_steps;
    state->causal_relation_participating_records =
        consensus.constraint_relation_participating_records;
    state->causal_relation_independent_sources =
        consensus.constraint_relation_independent_sources;
    state->causal_relation_source_contributions =
        consensus.constraint_relation_source_contributions;
    state->causal_relation_max_source_contribution =
        consensus.constraint_relation_max_source_contribution;
    state->causal_relation_contribution_concentration_q16 =
        consensus.constraint_relation_contribution_concentration_q16;
    state->causal_relation_singleton_supported_steps =
        consensus.constraint_relation_singleton_supported_steps;
    state->causal_relation_minimum_probe_support =
        consensus.constraint_relation_minimum_probe_support;
    state->causal_relation_component_digest =
        consensus.constraint_relation_component_digest;
    state->causal_relation_component_revision_digest =
        consensus.constraint_relation_component_revision_digest;
    state->causal_relation_external_provenance_digest =
        consensus.constraint_relation_external_provenance_digest;
    state->causal_relation_external_leaves =
        consensus.constraint_relation_external_leaves;
    state->causal_relation_trajectory_revision =
        consensus.constraint_relation_trajectory_revision;
  }
  return true;
}

BCC32_REWRITE_HD inline bool advance_resident_program_once(
    ResidentRewriteEngine engine,
    bool preserve_yielded_history = false) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  clear_generated_word(state);
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  Record& trajectory = state->records[trajectory_slot];
  if (trajectory.lane[4] == 0u) return false;

  const bool cursor_present =
      find_span_execution_cursor(state, trajectory.lane[1]) != kInvalid ||
      span_execution_cursor_collision(state, trajectory.lane[1]);
  if (cursor_present)
    return advance_span_execution_cursor(state, trajectory_slot);

  ProgramCandidateConsensus consensus{};
  collect_version_space_program_candidates(state, trajectory, &consensus);
  collect_resident_span_program_candidates(state, trajectory, &consensus);
  bool saw_unbound = false;
  collect_fixed_program_candidates(state, trajectory, &consensus,
                                   &saw_unbound);
  RawSpanConsensusBindingReceipt raw_span_binding{};
  const RawSpanCompositionCandidate raw_span_candidate =
      collect_raw_span_yielded_binding_candidate(state, trajectory);
  if (raw_span_candidate.have_candidate && !raw_span_candidate.conflict)
    bind_raw_span_composition_consensus(
        state, trajectory, raw_span_candidate, &consensus,
        &raw_span_binding);

  // Distributed closure is a resident authority, but a generalized
  // reafferent closure must not preempt an already-earned exact bootstrap
  // Program on the initial two-word query. Otherwise a learned returned
  // consequence replaces the organism's first physical action instead of
  // becoming the later consequence of that action. Exact context-linked
  // closure may still supersede the bootstrap candidate; an unlinked generic
  // closure waits until no ordinary candidate owns the query.
  const CausalRelationCandidate relation_probe =
      collect_causal_relation_candidate(state, trajectory);
  // An applicable but unresolved resident relation is evidence, not an
  // authority boundary. It must not prevent an independently grounded
  // Program/Span surface from producing the physical action whose returned
  // consequence can close the missing source component. A linked ambiguity
  // still fails closed without adding a semantic answer cell, while the
  // relation reader keeps unrelated generic relations out of this merge.
  if (relation_probe.applicable &&
      (!relation_probe.ready || relation_probe.ambiguous))
    collect_resident_causal_relation_candidate(state, trajectory, &consensus);
  const bool relation_owns_query =
      relation_probe.applicable && relation_probe.ready &&
      !relation_probe.ambiguous &&
      (relation_probe.query_linked || !consensus.have_candidate ||
       causal_relation_one_generated_tail(state, trajectory));
  if (relation_owns_query) {
    ProgramCandidateConsensus relation_consensus{};
    collect_resident_causal_relation_candidate(state, trajectory,
                                                &relation_consensus);
    if (!relation_consensus.have_candidate || relation_consensus.conflict ||
        !emit_program_candidate_word(state, relation_consensus)) {
      if (preserve_yielded_history) {
        trajectory.lane[4] = 0u;
        ++trajectory.revision;
      } else if ((trajectory.lane[7] & kTrajectoryHasGenerated) != 0u) {
        clear_trajectory(state, trajectory_slot);
      }
      refresh_receipt(state);
      return false;
    }
    refresh_receipt(state);
    return true;
  }
  if (!consensus.have_candidate && relation_probe.applicable &&
      relation_probe.ambiguous) {
    if (preserve_yielded_history) {
      trajectory.lane[4] = 0u;
      ++trajectory.revision;
    } else if ((trajectory.lane[7] & kTrajectoryHasGenerated) != 0u) {
      clear_trajectory(state, trajectory_slot);
    }
    refresh_receipt(state);
    return false;
  }
  if (saw_unbound) ++state->rejected_unbound_variables;
  if (consensus.span_saw_unbound)
    ++state->span_rejected_unbound_variables;
  if (consensus.span_saw_ambiguous)
    ++state->span_ambiguous_abstentions;
  if (!consensus.have_candidate || consensus.conflict || saw_unbound ||
      consensus.span_saw_unbound || consensus.span_saw_ambiguous) {
    if (consensus.conflict) {
      ++state->program_conflict_abstentions;
      if (consensus.version_space_saw_program)
        ++state->version_space_conflict_abstentions;
      if (consensus.span_saw_program)
        ++state->span_conflict_abstentions;
    }
    // Most direct callers retire a yielded generated history once no lawful
    // continuation remains. The mixed-provenance membrane instead preserves
    // the same grounded history for a later physical consequence. Keep that
    // distinction inside this single complete candidate pass: duplicating a
    // partial preflight in the caller both inflates the CUDA frame and can
    // reject newer candidate families (such as raw-span composition) before
    // they reach this canonical consensus.
    if (preserve_yielded_history) {
      trajectory.lane[4] = 0u;
      ++trajectory.revision;
    } else if ((trajectory.lane[7] & kTrajectoryHasGenerated) != 0u) {
      clear_trajectory(state, trajectory_slot);
    }
    refresh_receipt(state);
    return false;
  }
  if (raw_span_binding.bound && !raw_span_binding.conflict &&
      consensus.selected_from_span && consensus.span_contributed &&
      consensus.word == raw_span_binding.predicted_word &&
      consensus.extent == trajectory.lane[2] &&
      consensus.diagnostic_locus == raw_span_binding.upper_slot) {
    if (!emit_raw_span_consensus_word(state, trajectory, consensus,
                                      &raw_span_binding))
      return false;
    settle_program_candidates(state, trajectory);
    refresh_receipt(state);
    return true;
  }
  if (!emit_program_candidate_word(state, consensus)) return false;
  const bool cached_span =
      consensus.span_contributed &&
      install_span_execution_cursor(state, trajectory, consensus);
  if (!cached_span) settle_program_candidates(state, trajectory);
  refresh_receipt(state);
  return true;
}

// One admitted physical event advances the resident world exactly once. A
// quiet event is explicit admitted boundary matter (valid=0); host wall-clock
// gaps and polling never close an episode.
