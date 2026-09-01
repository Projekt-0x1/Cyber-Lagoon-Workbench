// Read one ordinary yielded raw query through the resident causal-constraint
// participation ecology.  This file is included inside the canonical
// causal_rewrite namespace after ProgramCandidateConsensus and its merge law.
// It owns no output register: a qualified result enters that consensus and the
// existing Program emitter remains the only generated-word writer.

namespace source_witness = resident_causal_relation_source_witness;

// mixed_provenance is textually defined after causal_rewrite_universe includes
// this reader.  Keep these narrow declarations here so the suffix-restart
// boundary can reuse the canonical provenance writer without duplicating its
// lane ABI or introducing a second stamping implementation.
namespace mixed_provenance {
enum class Origin : std::uint32_t;
BCC32_REWRITE_HD bool origin_at(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index, Origin* origin, std::uint32_t* producer);
BCC32_REWRITE_HD void clear_provenance(ResidentRewriteState* state,
                                        std::uint32_t owner);
BCC32_REWRITE_HD bool consume_external_event(
    ResidentRewriteEngine engine, RawRewriteEvent event, bool update_receipt);
}  // namespace mixed_provenance

#if defined(__CUDACC__)
#define BCC32_RELATION_READER_NOINLINE __noinline__
#else
#define BCC32_RELATION_READER_NOINLINE
#endif

struct CausalRelationCandidate {
  std::uint32_t word = 0u;
  std::uint32_t extent = 0u;
  // Passive readiness-stage receipt: 1=applicable outcome, 2=selected
  // outcome, 3=complete source identity, 4=record-cut closure,
  // 5=source provenance, 6=query provenance, 7=live-world resolution,
  // 8=ready. It never authorizes a write.
  std::uint32_t readiness_stage = 0u;
  // Passive source-provenance failure receipt: 1=invalid arguments,
  // 2=duplicate source trajectory, 3=source trajectory missing,
  // 4=source trajectory shape/flags, 5=provenance block missing,
  // 6=provenance content mismatch, 7=leaf-count mismatch.
  std::uint32_t source_provenance_failure = 0u;
  std::uint32_t source_provenance_owner = kInvalid;
  std::uint32_t trajectory_owner = kInvalid;
  std::uint32_t trajectory_revision = 0u;
  std::uint32_t relation = 0u;
  std::uint32_t positive_sources = 0u;
  std::uint32_t source_revision = 0u;
  std::uint32_t reafferent_sources = 0u;
  std::uint32_t ordinary_sources = 0u;
  std::uint32_t probe_steps = 0u;
  std::uint32_t participating_records = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t source_contributions = 0u;
  std::uint32_t maximum_source_contribution = 0u;
  std::uint32_t contribution_concentration_q16 = 0u;
  std::uint32_t singleton_supported_steps = 0u;
  std::uint32_t minimum_probe_support = 0u;
  std::uint64_t component_digest = 0u;
  std::uint64_t component_revision_digest = 0u;
  // Source-pair authority is external-only. Query integrity is separate and
  // may contain a generated/distributed resident context event.
  std::uint64_t source_provenance_digest = 0u;
  std::uint32_t source_external_leaves = 0u;
  std::uint64_t query_context_digest = 0u;
  std::uint32_t query_context_leaves = 0u;
  // Compatibility observation receipt: source plus query leaves/digest. This
  // remains available to existing reafference telemetry, but never authorizes
  // a relation emission.
  std::uint64_t external_provenance_digest = 0u;
  std::uint32_t external_leaves = 0u;
  bool applicable = false;
  bool ready = false;
  bool ambiguous = false;
  // A generalized relation may be residently mature without being connected
  // to this query's antecedent.  That distinction matters when another
  // exact candidate already owns the same (antecedent, relation) probe.
  bool query_linked = false;
};

// Scratch storage is an implementation aperture, not a semantic quorum. If
// a resident relation exceeds this bounded query workspace it abstains rather
// than growing the device stack or silently selecting a subset of sources.
inline constexpr std::uint32_t kRelationCandidateScratchCapacity = 32u;

BCC32_REWRITE_HD inline bool causal_participation_record(
    const Record& record) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormConstraintParticipation &&
         (record.lane[1] == 1u || record.lane[1] == 2u) &&
         record.lane[2] != 0u && record.lane[2] != kInvalid &&
         record.lane[3] != 0u && record.lane[3] != kInvalid &&
         record.lane[4] != 0u && record.lane[4] != kInvalid &&
         ((record.lane[5] == 1u && record.lane[6] == 0u) ||
          (record.lane[5] == 0u && record.lane[6] == 1u)) &&
         record.lane[7] == 0u && record.reserved[0] == 0u &&
         record.reserved[1] == 0u;
}

// Returned consequences are learned by the dedicated reafferent transaction,
// but they remain resident causal matter: after independent returns have
// matured a closure, later language/action contact must be able to reuse it.
// The witness marker is therefore a source-development receipt, not an
// exclusion flag. The reader gives a mature reafferent closure precedence over
// an older bootstrap consequence only when the live component itself proves
// that maturity; one returned contact can never manufacture an answer.
BCC32_REWRITE_HD inline bool language_relation_source_eligible(
    const ResidentRewriteState* state, const Record& record) {
  (void)state;
  return causal_participation_record(record);
}

// A relation whose sources are all reafferent-witnessed (i.e. each source
// carries a resident claim bound to a live world cell) must be reread from
// that live world state at answer time, never from the value captured when
// the relation was originally assimilated. This is the read-side half of the
// turn/world-consequence binding: the write side stages a subject-scoped
// claim per accepted source revision (stage_device_body_world_consequence),
// this reads it back through the same claim/binding pair. If sources
// disagree on the live world value, or any claim/binding has been severed,
// the candidate is unusable rather than silently falling back to a stale
// cached word.
struct CausalRelationLiveWorldResolution {
  bool bound = false;
  bool valid = true;
  std::uint32_t word = kInvalid;
};
BCC32_REWRITE_HD inline CausalRelationLiveWorldResolution
causal_relation_live_world_value(
    const ResidentRewriteState* state,
    const std::uint32_t* sources, std::uint32_t source_count) {
  CausalRelationLiveWorldResolution result{};
  if (state == nullptr || sources == nullptr || source_count == 0u)
    return result;
  std::uint32_t reafferent_sources = 0u;
  std::uint32_t claimed_sources = 0u;
  std::uint32_t common_world_owner = kInvalid;
  std::uint32_t current_word = kInvalid;
  for (std::uint32_t index = 0u; index < source_count; ++index) {
    const std::uint32_t source = sources[index];
    if (!source_witness::is_reafferent_witness(state, source))
      continue;
    ++reafferent_sources;
    const turn_world_consequence_binding::SourceClaimResolution live =
        turn_world_consequence_binding::
            resolve_source_claim_current_value(state, source);
    if (!live.present)
      continue;
    ++claimed_sources;
    result.bound = true;
    if (!live.valid || live.ambiguous) {
      result.valid = false;
      return result;
    }
    if (common_world_owner == kInvalid) {
      common_world_owner = live.world_owner;
      current_word = live.value;
    } else if (live.world_owner != common_world_owner ||
               live.value != current_word) {
      result.valid = false;
      return result;
    }
  }
  if (!result.bound)
    return result;
  if (reafferent_sources == 0u || claimed_sources != reafferent_sources) {
    result.valid = false;
    return result;
  }
  result.word = current_word;
  return result;
}

// Defined below with the full provenance check; the source-withdrawal reader
// is kept as a separate device-call boundary so its scan does not accumulate
// with the relation candidate's scratch workspace.
BCC32_REWRITE_HD inline bool causal_relation_one_generated_tail(
    const ResidentRewriteState* state, const Record& trajectory);

// Reafferent promotion is source-withdrawal sensitive only to the raw source
// ecology that actually taught the bootstrap consequence just emitted on this
// trajectory. A global scan of every closed trajectory turns unrelated
// contacts, withdrawal sentinels, and autonomous bootstrap history into veto
// authority over this relation. The participation pair already carries the
// resident source identity, so use that topology and no host/semantic label.
static BCC32_REWRITE_HD BCC32_RELATION_READER_NOINLINE bool
reafferent_bootstrap_source_free(const ResidentRewriteState* state,
                                 const Record& query,
                                 std::uint32_t relation) {
  if (state == nullptr || relation == 0u || relation == kInvalid ||
      !causal_relation_one_generated_tail(state, query))
    return false;

  std::uint32_t bootstrap_after = 0u;
  if (!trajectory_word_at(state, query.lane[1], 2u, &bootstrap_after) ||
      bootstrap_after == 0u || bootstrap_after == kInvalid)
    return false;

  bool saw_bootstrap_source = false;
  for (std::uint32_t participant_slot = 0u;
       participant_slot < live_record_capacity(state); ++participant_slot) {
    const Record& participant = state->records[participant_slot];
    if (!causal_participation_record(participant) ||
        participant.lane[1] != 2u || participant.lane[2] != relation ||
        participant.lane[3] != bootstrap_after || participant.lane[5] != 1u ||
        participant.lane[6] != 0u ||
        source_witness::is_reafferent_witness(state, participant.lane[4]))
      continue;

    saw_bootstrap_source = true;
    for (std::uint32_t source_slot = 0u;
         source_slot < live_record_capacity(state); ++source_slot) {
      const Record& source = state->records[source_slot];
      if (source.matter_q8 != 0u && source.lane[0] == kFormTrajectory &&
          source.lane[1] == participant.lane[4] && source.lane[3] != 0u &&
          (source.lane[7] & kTrajectoryHasGenerated) == 0u)
        return false;
    }
  }
  return saw_bootstrap_source;
}

BCC32_REWRITE_HD inline bool exact_causal_participation_pair(
    const ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after,
    std::uint32_t source_revision, bool positive,
    std::uint32_t* consequent_slot) {
  if (consequent_slot != nullptr) *consequent_slot = kInvalid;
  if (state == nullptr || before == 0u || before == kInvalid ||
      relation == 0u || relation == kInvalid || after == 0u ||
      after == kInvalid || source_revision == 0u ||
      source_revision == kInvalid)
    return false;
  std::uint32_t antecedents = 0u;
  std::uint32_t consequents = 0u;
  std::uint32_t source_fragments = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (!language_relation_source_eligible(state, record) ||
        record.lane[4] != source_revision || record.lane[2] != relation)
      continue;
    ++source_fragments;
    if (record.lane[5] != (positive ? 1u : 0u) ||
        record.lane[6] != (positive ? 0u : 1u))
      return false;
    if (record.lane[1] == 1u && record.lane[3] == before)
      ++antecedents;
    if (record.lane[1] == 2u && record.lane[3] == after) {
      ++consequents;
      if (consequent_slot != nullptr) *consequent_slot = slot;
    }
  }
  return source_fragments == 2u && antecedents == 1u && consequents == 1u;
}

BCC32_REWRITE_HD inline bool source_already_counted(
    const std::uint32_t sources[kRecordCapacity], std::uint32_t count,
    std::uint32_t source) {
  for (std::uint32_t index = 0u; index < count; ++index)
    if (sources[index] == source) return true;
  return false;
}

struct CausalRelationStepCandidate {
  std::uint32_t after = 0u;
  std::uint32_t positive_sources = 0u;
  std::uint32_t counter_sources = 0u;
  std::uint32_t minimum_source = kInvalid;
  std::uint32_t maximum_source = 0u;
  bool cut_closed = false;
};

BCC32_REWRITE_HD inline bool causal_relation_step_less(
    const CausalRelationStepCandidate& left,
    const CausalRelationStepCandidate& right) {
  if (left.after != right.after) return left.after < right.after;
  if (left.minimum_source != right.minimum_source)
    return left.minimum_source < right.minimum_source;
  return left.maximum_source < right.maximum_source;
}

BCC32_REWRITE_HD inline bool causal_relation_pair_exists_excluding_record(
    const ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after,
    std::uint32_t excluded_slot) {
  if (state == nullptr) return false;
  const std::uint32_t excluded_source =
      excluded_slot < live_record_capacity(state) ? state->records[excluded_slot].lane[4]
                                      : kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (slot == excluded_slot || !language_relation_source_eligible(state, record) ||
        record.lane[1] != 2u || record.lane[2] != relation ||
        record.lane[3] != after || record.lane[5] == 0u ||
        record.lane[4] == excluded_source)
      continue;
    std::uint32_t consequent_slot = kInvalid;
    if (exact_causal_participation_pair(
            state, before, relation, after, record.lane[4], true,
            &consequent_slot) &&
        consequent_slot != excluded_slot)
      return true;
  }
  return false;
}

// Authority is the intervention law over the entire matching component: every
// participating physical fragment can be removed while another complete path
// still carries the same transition. No source count or semantic quorum enters
// selection; the amount of redundancy is whatever resident history grew.
BCC32_REWRITE_HD inline bool causal_relation_component_survives_record_cuts(
    const ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after) {
  bool saw_component = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (!language_relation_source_eligible(state, record) ||
        record.lane[2] != relation ||
        record.lane[5] == 0u)
      continue;
    std::uint32_t consequent_slot = kInvalid;
    const bool pair = exact_causal_participation_pair(
        state, before, relation, after, record.lane[4], true,
        &consequent_slot);
    if (!pair)
      continue;
    saw_component = true;
    // Counterfactually retain this one physical participant alone. The
    // canonical fragment form must not encode both endpoints of a transition;
    // otherwise one localist Record could survive the later cut assay.
    const bool single_record_sufficient =
        record.lane[1] == 1u && record.lane[1] == 2u &&
        record.lane[3] == before && record.lane[3] == after;
    if (single_record_sufficient) return false;
    if (!causal_relation_pair_exists_excluding_record(
            state, before, relation, after, slot))
      return false;
  }
  return saw_component;
}

BCC32_REWRITE_HD inline bool
causal_relation_cross_context_pair_exists_excluding_record(
    const ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after, std::uint32_t excluded_slot) {
  if (state == nullptr || relation == 0u || relation == kInvalid ||
      after == 0u || after == kInvalid)
    return false;
  const std::uint32_t excluded_source =
      excluded_slot < live_record_capacity(state)
          ? state->records[excluded_slot].lane[4]
          : kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (slot == excluded_slot || !language_relation_source_eligible(state, record) ||
        record.lane[1] != 2u || record.lane[2] != relation ||
        record.lane[3] != after || record.lane[4] == excluded_source ||
        record.lane[5] != 1u || record.lane[6] != 0u)
      continue;
    std::uint32_t matching_fragments = 0u;
    std::uint32_t antecedents = 0u;
    std::uint32_t consequents = 0u;
    for (std::uint32_t peer_slot = 0u;
         peer_slot < live_record_capacity(state); ++peer_slot) {
      const Record& peer = state->records[peer_slot];
      if (!language_relation_source_eligible(state, peer) ||
          peer.lane[4] != record.lane[4] || peer.lane[2] != relation ||
          peer.lane[5] != 1u || peer.lane[6] != 0u)
        continue;
      ++matching_fragments;
      if (peer.lane[1] == 1u) {
        ++antecedents;
      } else if (peer.lane[1] == 2u && peer.lane[3] == after) {
        ++consequents;
      }
    }
    if (matching_fragments == 2u && antecedents == 1u && consequents == 1u)
      return true;
  }
  return false;
}

// Generic transfer checks the complete positive component across the
// independently sourced antecedents that actually taught this consequence.
// The current query may be novel, so its first word is not an antecedent of
// those historical source pairs.  Derive each source's own antecedent and
// apply the same one-record cut law without importing query-local content.
BCC32_REWRITE_HD inline bool
causal_relation_cross_context_component_survives_record_cuts(
    const ResidentRewriteState* state, std::uint32_t relation,
    std::uint32_t after) {
  if (state == nullptr || relation == 0u || relation == kInvalid ||
      after == 0u || after == kInvalid)
    return false;
  bool saw_component = false;
  bool saw_reafferent = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (!language_relation_source_eligible(state, record) ||
        record.lane[2] != relation || record.lane[5] != 1u ||
        record.lane[6] != 0u)
      continue;
    const std::uint32_t source = record.lane[4];
    std::uint32_t matching_fragments = 0u;
    std::uint32_t antecedents = 0u;
    std::uint32_t consequents = 0u;
    for (std::uint32_t peer_slot = 0u;
         peer_slot < live_record_capacity(state); ++peer_slot) {
      const Record& peer = state->records[peer_slot];
      if (!language_relation_source_eligible(state, peer) ||
          peer.lane[4] != source || peer.lane[2] != relation ||
          peer.lane[5] != 1u || peer.lane[6] != 0u)
        continue;
      ++matching_fragments;
      if (peer.lane[1] == 1u) {
        ++antecedents;
      } else if (peer.lane[1] == 2u && peer.lane[3] == after) {
        ++consequents;
      }
    }
    if (matching_fragments != 2u || antecedents != 1u || consequents != 1u)
      continue;
    saw_component = true;
    saw_reafferent |= source_witness::is_reafferent_witness(state, source);
    if (!causal_relation_cross_context_pair_exists_excluding_record(
            state, relation, after, slot))
      return false;
  }
  return saw_component && saw_reafferent;
}

// A result is qualified by the component intervention above. Counterevidence
// or two separately cut-closed outcomes remains ambiguous and silent.
static BCC32_REWRITE_HD BCC32_RELATION_READER_NOINLINE
CausalRelationStepCandidate
collect_causal_relation_step_candidate(
    const ResidentRewriteState* state, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after) {
  CausalRelationStepCandidate candidate{};
  candidate.after = after;
  if (state == nullptr) return candidate;
  // The resident aperture is a scan bound, not a reason to reserve one
  // full-record array on every CUDA relation probe.  A probe that would
  // exceed this bounded scratch is deliberately ineligible rather than
  // growing the device stack or silently selecting a subset.
  std::uint32_t positive[kRelationCandidateScratchCapacity]{};
  std::uint32_t counter[kRelationCandidateScratchCapacity]{};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (!language_relation_source_eligible(state, record) ||
        record.lane[1] != 2u ||
        record.lane[2] != relation || record.lane[3] != after)
      continue;
    const bool is_positive = record.lane[5] == 1u;
    if (!exact_causal_participation_pair(
            state, before, relation, after, record.lane[4], is_positive,
            nullptr))
      continue;
    std::uint32_t* sources = is_positive ? positive : counter;
    std::uint32_t& count = is_positive ? candidate.positive_sources
                                       : candidate.counter_sources;
    if (source_already_counted(sources, count, record.lane[4])) continue;
    if (count == kRelationCandidateScratchCapacity) {
      candidate.cut_closed = false;
      return candidate;
    }
    sources[count++] = record.lane[4];
    if (is_positive &&
        (candidate.minimum_source == kInvalid ||
         record.lane[4] < candidate.minimum_source)) {
      candidate.minimum_source = record.lane[4];
    }
    if (is_positive && record.lane[4] > candidate.maximum_source)
      candidate.maximum_source = record.lane[4];
  }
  candidate.cut_closed = candidate.counter_sources == 0u &&
      causal_relation_component_survives_record_cuts(
          state, before, relation, after);
  return candidate;
}

BCC32_REWRITE_HD inline std::uint64_t causal_relation_mix(
    std::uint64_t digest, std::uint64_t value) {
  digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) +
            (digest >> 2u);
  digest *= 0xbf58476d1ce4e5b9ull;
  return digest;
}

BCC32_REWRITE_HD inline std::uint64_t causal_relation_fragment_digest(
    const Record& record, std::uint32_t probe, bool revision_only) {
  std::uint64_t digest = revision_only ? 0x94d049bb133111ebull
                                       : 0x243f6a8885a308d3ull;
  digest = causal_relation_mix(digest, probe);
  if (revision_only) {
    digest = causal_relation_mix(digest, record.lane[4]);
    digest = causal_relation_mix(digest, record.revision);
    digest = causal_relation_mix(digest, record.matter_q8);
    return digest;
  }
  for (std::uint32_t lane = 0u; lane < kLaneCount; ++lane)
    digest = causal_relation_mix(digest, record.lane[lane]);
  digest = causal_relation_mix(digest, record.reserved[0]);
  digest = causal_relation_mix(digest, record.reserved[1]);
  digest = causal_relation_mix(digest, record.matter_q8);
  return digest;
}

// Fold the live physical origin of one participation source. A stale owner,
// duplicate source trajectory, missing provenance block, or any generated
// leaf makes the complete closure ineligible. The digest is only a locator:
// publication repeats these exact validations and the component cut assay.
BCC32_REWRITE_HD inline bool causal_relation_retained_source_word_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr) return false;
  const std::uint32_t ordinal = index / 2u;
  std::uint32_t term_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormTrajectoryTerm ||
        term.lane[1] != owner || term.lane[2] != ordinal)
      continue;
    if (term_slot != kInvalid) return false;
    term_slot = slot;
  }
  if (term_slot == kInvalid) return false;
  *word = state->records[term_slot].lane[4u + (index % 2u)];
  return *word != 0u && *word != kInvalid;
}

// Read one provenance leaf without importing the mixed-provenance ingress
// adapter into this textually included reader. The lane ABI is canonical
// resident matter; only the two non-authoritative origins accepted below are
// interpreted here.
BCC32_REWRITE_HD inline bool causal_relation_query_origin_at(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index, std::uint32_t* origin, std::uint32_t* producer) {
  if (origin != nullptr) *origin = kInvalid;
  if (producer != nullptr) *producer = kInvalid;
  if (state == nullptr || origin == nullptr || producer == nullptr ||
      index >= trajectory.lane[2])
    return false;
  const std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryProvenance, trajectory.lane[1], index / 2u);
  if (slot == kInvalid) return false;
  const Record& provenance = state->records[slot];
  const std::uint32_t local = index % 2u;
  if (provenance.matter_q8 == 0u ||
      provenance.lane[0] != kFormTrajectoryProvenance ||
      (provenance.lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
      (provenance.lane[kProvenanceValidityLane] & (1u << local)) == 0u)
    return false;
  const std::uint32_t origin_lane = 3u + local * 2u;
  const std::uint32_t producer_lane = origin_lane + 1u;
  *origin = provenance.lane[origin_lane];
  *producer = provenance.lane[producer_lane];
  if (*origin == kProvenanceExternalOrigin ||
      *origin == kProvenanceDistributedOrigin)
    return *producer == kInvalid;
  if (*origin == kProvenanceGeneratedOrigin)
    // Historical producer identity is integrity context, not current
    // authority. The producer may later be lesioned; source-pair teaching
    // remains external-only below.
    return *producer != kInvalid;
  return false;
}

// A generated or distributed resident event may become context for a later
// raw relation. This helper recognizes provenance shape, not content: all
// events are still raw resident words, and only an immediately following fresh
// external event can reopen the relation reader. Context contributes no source
// teaching authority.
BCC32_REWRITE_HD inline bool causal_relation_distributed_context_suffix(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t* context_index, std::uint32_t* relation_index) {
  if (context_index != nullptr) *context_index = kInvalid;
  if (relation_index != nullptr) *relation_index = kInvalid;
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u)
    return false;

  bool saw_context = false;
  bool saw_external_after = false;
  std::uint32_t last_context = kInvalid;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t origin = kProvenanceExternalOrigin;
    std::uint32_t producer = kInvalid;
    if (!causal_relation_query_origin_at(
            state, trajectory, index, &origin, &producer))
      return false;
    if (origin == kProvenanceDistributedOrigin ||
        origin == kProvenanceGeneratedOrigin) {
      if (origin == kProvenanceDistributedOrigin && producer != kInvalid)
        return false;
      if (origin == kProvenanceGeneratedOrigin && producer == kInvalid)
        return false;
      saw_context = true;
      saw_external_after = false;
      last_context = index;
      continue;
    }
    if (origin == kProvenanceExternalOrigin) {
      if (producer != kInvalid) return false;
      if (saw_context) saw_external_after = true;
      continue;
    }
    return false;
  }
  if (!saw_context || !saw_external_after || last_context == kInvalid ||
      last_context + 1u >= trajectory.lane[2])
    return false;
  std::uint32_t relation_origin = kProvenanceExternalOrigin;
  std::uint32_t relation_producer = kInvalid;
  if (!causal_relation_query_origin_at(
      state, trajectory, last_context + 1u, &relation_origin,
          &relation_producer) ||
      relation_origin != kProvenanceExternalOrigin ||
      relation_producer != kInvalid)
    return false;
  if (context_index != nullptr) *context_index = last_context;
  if (relation_index != nullptr) *relation_index = last_context + 1u;
  return true;
}

// A generated Carry is valid context only while the following external
// suffix actually engages the resident relation ecology.  If PAUSE proves
// that no relation is applicable, do not let that exhausted context poison a
// wholly ordinary next query: preserve the exact external suffix as a fresh
// trajectory and retire only the zero-authority generated prefix.  This is a
// bounded resident reshaping operation; it does not select a word, infer a
// relation, or manufacture evidence.
#if defined(__CUDACC__)
__device__ __noinline__
#else
inline
#endif
bool restart_external_suffix_after_unresolved_context(
    ResidentRewriteState* state);

// A trajectory whose LAST word is a genuine, fully-tagged distributed or
// generated event -- not merely any HasGenerated trajectory -- is worth
// preserving as zero-authority execution context across a physical END
// boundary. Unlike causal_relation_distributed_context_suffix (which also
// requires a trailing external word, appropriate for validating a live
// query), this checks the trajectory at the moment its context was JUST
// derived, before any further contact has arrived at all.
BCC32_REWRITE_HD inline bool causal_relation_trajectory_ends_with_context(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u ||
      trajectory.lane[2] == 0u)
    return false;
  const std::uint32_t last_index = trajectory.lane[2] - 1u;
  std::uint32_t origin = kProvenanceExternalOrigin;
  std::uint32_t producer = kInvalid;
  if (!causal_relation_query_origin_at(state, trajectory, last_index, &origin,
                                       &producer))
    return false;
  if (origin == kProvenanceDistributedOrigin) return producer == kInvalid;
  if (origin == kProvenanceGeneratedOrigin) return producer != kInvalid;
  return false;
}

// A bootstrap Program may first emit an ordinary intermediate consequence.
// That generated tail is not evidence and cannot teach the relation. It only
// leaves the original two external query words resident so a separately
// grounded distributed consequence can complete the same physical trajectory.
// Keep this aperture to exactly one generated word; after the distributed
// consequence is appended the adult cannot loop the same relation forever.
BCC32_REWRITE_HD inline bool causal_relation_one_generated_tail(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] != 3u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u)
    return false;
  for (std::uint32_t index = 0u; index < 3u; ++index) {
    const std::uint32_t provenance_slot = find_owned_block(
        state, kFormTrajectoryProvenance, trajectory.lane[1], index / 2u);
    if (provenance_slot == kInvalid) return false;
    const Record& provenance = state->records[provenance_slot];
    const std::uint32_t local = index % 2u;
    const std::uint32_t origin_lane = 3u + local * 2u;
    const std::uint32_t producer_lane = origin_lane + 1u;
    if ((provenance.lane[kProvenanceValidityLane] & (1u << local)) == 0u)
      return false;
    const std::uint32_t origin = provenance.lane[origin_lane];
    const std::uint32_t producer = provenance.lane[producer_lane];
    if (index < 2u) {
      if (origin != kProvenanceExternalOrigin || producer != kInvalid)
        return false;
    } else if (origin != kProvenanceGeneratedOrigin ||
               producer == kInvalid) {
      return false;
    }
  }
  return true;
}

BCC32_REWRITE_HD inline bool causal_relation_external_source_provenance(
    const ResidentRewriteState* state, std::uint32_t source_owner,
    std::uint64_t* source_digest, std::uint32_t* external_leaves,
    std::uint32_t* failure_code = nullptr) {
  if (failure_code != nullptr) *failure_code = 0u;
  if (state == nullptr || source_digest == nullptr ||
      external_leaves == nullptr || source_owner == 0u ||
      source_owner == kInvalid) {
    if (failure_code != nullptr) *failure_code = 1u;
    return false;
  }
  // Source pages are deliberately withdrawable. A compact resident witness
  // is the only lawful handoff after that withdrawal; it was created while
  // the complete external source still existed. The witness alone is never
  // source authority: at least one live causal-participation record must still
  // reference this exact owner, otherwise a retained/stale witness cannot
  // qualify an answer.
  bool source_still_participates = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& participant = state->records[slot];
    if (causal_participation_record(participant) &&
        participant.lane[4] == source_owner) {
      source_still_participates = true;
      break;
    }
  }
  if (source_still_participates &&
      source_witness::witness_valid(
          state, source_owner, source_digest, external_leaves))
    return true;
  std::uint32_t source_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormTrajectory ||
        candidate.lane[1] != source_owner)
      continue;
    if (source_slot != kInvalid) {
      if (failure_code != nullptr) *failure_code = 2u;
      return false;
    }
    source_slot = slot;
  }
  if (source_slot == kInvalid) {
    if (failure_code != nullptr) *failure_code = 3u;
    return false;
  }
  const Record& source = state->records[source_slot];
  if (source.lane[2] < 3u || source.lane[3] == 0u ||
      (source.lane[7] & kTrajectoryHasGenerated) != 0u) {
    if (failure_code != nullptr) *failure_code = 4u;
    return false;
  }
  std::uint64_t digest = causal_relation_mix(
      0x13198a2e03707344ull, source.lane[1]);
  digest = causal_relation_mix(digest, source.revision);
  std::uint32_t leaves = 0u;
  for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
    const std::uint32_t provenance_slot = find_owned_block(
        state, kFormTrajectoryProvenance, source.lane[1], index / 2u);
    if (provenance_slot == kInvalid) {
      if (failure_code != nullptr) *failure_code = 5u;
      return false;
    }
    const Record& provenance = state->records[provenance_slot];
    const std::uint32_t local = index % 2u;
    const std::uint32_t origin_lane = 3u + local * 2u;
    const std::uint32_t producer_lane = origin_lane + 1u;
    const std::uint32_t valid_bit = 1u << local;
    std::uint32_t word = 0u;
    if ((provenance.lane[kProvenanceValidityLane] & valid_bit) == 0u ||
        provenance.lane[origin_lane] != kProvenanceExternalOrigin ||
        provenance.lane[producer_lane] != kInvalid ||
        !causal_relation_retained_source_word_at(
            state, source.lane[1], index, &word)) {
      if (failure_code != nullptr) *failure_code = 6u;
      return false;
    }
    digest = causal_relation_mix(digest, index);
    digest = causal_relation_mix(digest, word);
    digest = causal_relation_mix(digest, provenance.revision);
    digest = causal_relation_mix(digest, provenance.matter_q8);
    ++leaves;
  }
  *source_digest = digest;
  *external_leaves = leaves;
  if (leaves != source.lane[2]) {
    if (failure_code != nullptr) *failure_code = 7u;
    return false;
  }
  return true;
}

BCC32_REWRITE_HD inline bool causal_relation_external_query_provenance(
    const ResidentRewriteState* state, const Record& query,
    std::uint64_t* query_digest, std::uint32_t* external_leaves,
    std::uint32_t* query_leaves) {
  if (state == nullptr || query_digest == nullptr ||
      external_leaves == nullptr || query_leaves == nullptr ||
      query.matter_q8 == 0u ||
      query.lane[0] != kFormTrajectory ||
      (query.lane[3] != 0u &&
       (query.lane[7] & kTrajectoryOpenInquiry) == 0u) ||
      (query.lane[7] &
       ~(kTrajectoryWasYielded | kTrajectoryOpenInquiry |
         kTrajectoryHasGenerated | kTrajectoryHasCarry)) != 0u ||
      // A generated-execution Carry (a distributed/generated bridge that
      // survived a physical END as zero-authority context) is eligible;
      // an ordinary external-prefix Carry, with no generated bit at all,
      // is not relation-query matter.
      ((query.lane[7] & kTrajectoryHasCarry) != 0u &&
       (query.lane[7] & kTrajectoryHasGenerated) == 0u))
    return false;
  const bool distributed_context =
      causal_relation_distributed_context_suffix(state, query, nullptr,
                                                 nullptr);
  const bool ordinary_generated_tail =
      causal_relation_one_generated_tail(state, query);
  if ((query.lane[7] & kTrajectoryHasGenerated) != 0u &&
      !distributed_context && !ordinary_generated_tail)
    return false;
  std::uint64_t digest = causal_relation_mix(
      0xa4093822299f31d0ull, query.lane[1]);
  digest = causal_relation_mix(digest, query.revision);
  std::uint32_t leaves = 0u;
  for (std::uint32_t index = 0u; index < query.lane[2]; ++index) {
    const std::uint32_t provenance_slot = find_owned_block(
        state, kFormTrajectoryProvenance, query.lane[1], index / 2u);
    if (provenance_slot == kInvalid) return false;
    const Record& provenance = state->records[provenance_slot];
    const std::uint32_t local = index % 2u;
    const std::uint32_t origin_lane = 3u + local * 2u;
    const std::uint32_t producer_lane = origin_lane + 1u;
    std::uint32_t word = 0u;
    if ((provenance.lane[kProvenanceValidityLane] & (1u << local)) == 0u ||
        !trajectory_word_at(state, query.lane[1], index, &word))
      return false;
    const std::uint32_t origin = provenance.lane[origin_lane];
    const std::uint32_t producer = provenance.lane[producer_lane];
    if (origin == kProvenanceExternalOrigin) {
      if (producer != kInvalid) return false;
      ++leaves;
    } else if (origin == kProvenanceDistributedOrigin) {
      if (!distributed_context || producer != kInvalid) return false;
    } else if (origin == kProvenanceGeneratedOrigin) {
      if (ordinary_generated_tail && index == 2u) {
        // The bootstrap output is explicitly excluded from the query digest
        // and external-leaf count. It is a continuation trigger, never
        // evidence.
        continue;
      }
      // Ordinary generated context is integrity-bearing context only. It may
      // participate in the query digest after a live distributed context
      // suffix, but its producer never pays source authority.
      if (!distributed_context || producer == kInvalid) return false;
    } else {
      return false;
    }
    digest = causal_relation_mix(digest, index);
    digest = causal_relation_mix(digest, word);
    digest = causal_relation_mix(digest, origin);
    digest = causal_relation_mix(digest, producer);
    digest = causal_relation_mix(digest, provenance.revision);
    digest = causal_relation_mix(digest, provenance.matter_q8);
    ++*query_leaves;
  }
  *query_digest = digest;
  *external_leaves = leaves;
  // external_leaves remains compatibility telemetry for raw external query
  // words; query_leaves covers every live query/context word for integrity.
  return leaves != 0u;
}

BCC32_REWRITE_HD inline bool causal_relation_query_trajectory(
    const ResidentRewriteState* state, const Record& trajectory) {
  // The ordinary word-0/word-1 relation-chain reader below is coupled to a
  // fixed local probe_before[3]/probe_relation[3]/probe_after[3] aperture and
  // must keep its <=4-word cap. A distributed-context trajectory never falls
  // through into that reader (collect_causal_relation_candidate returns
  // before reaching it whenever causal_relation_distributed_context_suffix
  // succeeds), so it may use the ordinary physical trajectory capacity
  // instead -- unrelated raw contacts inserted between the resident bridge
  // and its eventual relation connective must not be rejected here purely
  // for exceeding a cap that was never about discourse length.
  const bool distributed_context = causal_relation_distributed_context_suffix(
      state, trajectory, nullptr, nullptr);
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] < 2u ||
      (!distributed_context && trajectory.lane[2] > 4u) ||
      (trajectory.lane[3] != 0u &&
       ((trajectory.lane[7] & kTrajectoryOpenInquiry) == 0u ||
        trajectory.lane[3] != 1u)) ||
      (trajectory.lane[4] != 0u &&
       (trajectory.lane[7] & kTrajectoryWasYielded) == 0u) ||
      (trajectory.lane[7] &
       ~(kTrajectoryWasYielded | kTrajectoryOpenInquiry |
         kTrajectoryHasGenerated | kTrajectoryHasCarry)) != 0u ||
      ((trajectory.lane[7] & kTrajectoryHasCarry) != 0u &&
       (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u) ||
      (trajectory.lane[7] & kTrajectoryWasYielded) == 0u ||
      trajectory.lane[1] == 0u || trajectory.lane[1] == kInvalid)
    return false;
  if ((trajectory.lane[7] & kTrajectoryHasGenerated) != 0u &&
      !causal_relation_distributed_context_suffix(state, trajectory, nullptr,
                                                  nullptr) &&
      !causal_relation_one_generated_tail(state, trajectory))
    return false;
  if (trajectory.lane[5] != kInvalid &&
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u)
    return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], index, &word) ||
        word == 0u || word == kInvalid)
      return false;
  }
  return true;
}

// A query may introduce a new antecedent while reusing a resident relation
// topology. In that case exact pair lookup is intentionally insufficient: the
// learned construction is the repeated relation -> consequence closure across
// independently sourced antecedents. Require at least two complete positive
// source pairs and a unique redundant consequence. A one-source or competing
// outcome remains applicable-but-unready and therefore abstains.
static BCC32_REWRITE_HD BCC32_RELATION_READER_NOINLINE
CausalRelationCandidate
collect_common_relation_candidate_filtered(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t relation, std::uint32_t antecedent_filter,
    std::uint32_t query_before_index = 0u) {
  CausalRelationCandidate result{};
  if (state == nullptr || relation == 0u || relation == kInvalid)
    return result;
  std::uint32_t query_before = 0u;
  if (!trajectory_word_at(state, trajectory.lane[1], query_before_index,
                          &query_before))
    return result;

  std::uint32_t outcomes[kRelationCandidateScratchCapacity]{};
  std::uint32_t outcome_sources[kRelationCandidateScratchCapacity]{};
  std::uint32_t outcome_reafferent_sources[kRelationCandidateScratchCapacity]{};
  std::uint32_t outcome_ordinary_sources[kRelationCandidateScratchCapacity]{};
  std::uint32_t outcome_count = 0u;
  // The resident record aperture bounds the total number of distinct
  // (outcome, source) pairs. Keep that relation in a linear table rather than
  // allocating a kRecordCapacity² local matrix on every CUDA query.
  std::uint32_t source_keys[kRelationCandidateScratchCapacity]{};
  std::uint32_t source_outcomes[kRelationCandidateScratchCapacity]{};
  std::uint32_t pair_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& consequent = state->records[slot];
    if (!language_relation_source_eligible(state, consequent) ||
        consequent.lane[1] != 2u || consequent.lane[2] != relation ||
        consequent.lane[5] != 1u)
      continue;
    std::uint32_t antecedent_count = 0u;
    std::uint32_t consequent_count = 0u;
    for (std::uint32_t peer_slot = 0u; peer_slot < live_record_capacity(state);
         ++peer_slot) {
      const Record& peer = state->records[peer_slot];
      if (!language_relation_source_eligible(state, peer) ||
          peer.lane[2] != relation || peer.lane[4] != consequent.lane[4])
        continue;
      if (peer.lane[5] != 1u || peer.lane[6] != 0u) return result;
      if (peer.lane[1] == 1u &&
          (antecedent_filter == 0u || peer.lane[3] == antecedent_filter))
        ++antecedent_count;
      if (peer.lane[1] == 1u && peer.lane[3] == query_before) {
        // An exact antecedent filter is already a live participation
        // fragment. It remains valid after its raw source page is withdrawn;
        // demanding source-word replay here would make exact context depend
        // on transcript retention. Generic (unfiltered) linkage keeps the
        // stricter retained-source witness check below.
        if (antecedent_filter != 0u) {
          result.query_linked = true;
        } else {
          std::uint32_t source_before = 0u;
          if (causal_relation_retained_source_word_at(
                  state, consequent.lane[4], 0u, &source_before) &&
              source_before == query_before)
            result.query_linked = true;
        }
      }
      if (peer.lane[1] == 2u && peer.lane[3] == consequent.lane[3])
        ++consequent_count;
    }
    if (antecedent_count != 1u || consequent_count != 1u) continue;

    std::uint32_t outcome = kInvalid;
    for (std::uint32_t index = 0u; index < outcome_count; ++index)
      if (outcomes[index] == consequent.lane[3]) outcome = outcomes[index];
    if (outcome == kInvalid) {
      if (outcome_count == kRelationCandidateScratchCapacity) return result;
      outcome = consequent.lane[3];
      outcomes[outcome_count] = outcome;
      outcome_sources[outcome_count] = 0u;
      ++outcome_count;
    }
    std::uint32_t outcome_index = 0u;
    while (outcomes[outcome_index] != outcome) ++outcome_index;
    bool source_seen = false;
    for (std::uint32_t index = 0u; index < pair_count; ++index)
      source_seen |= source_outcomes[index] == outcome &&
          source_keys[index] == consequent.lane[4];
    if (!source_seen) {
      if (pair_count == kRelationCandidateScratchCapacity) return result;
      source_outcomes[pair_count] = outcome;
      source_keys[pair_count] = consequent.lane[4];
      ++pair_count;
      ++outcome_sources[outcome_index];
      if (source_witness::is_reafferent_witness(
              state, consequent.lane[4]))
        ++outcome_reafferent_sources[outcome_index];
      else
        ++outcome_ordinary_sources[outcome_index];
    }
  }

  // Presence of a live candidate is only an applicability receipt. It is not
  // a resolved consequence: one-source support and equal competing outcomes
  // must remain visible to callers so they can abstain or learn to ask.
  result.applicable = outcome_count != 0u;
  result.readiness_stage = result.applicable ? 1u : 0u;
  std::uint32_t selected = kInvalid;
  std::uint32_t selected_sources = 0u;
  std::uint32_t selected_index = kInvalid;
  // A returned consequence is not a magical override. It becomes a mature
  // resident closure only when the live component survives its own physical
  // fragment cuts and the exact raw source ecology that taught the currently
  // emitted bootstrap route is gone. No fixed source quorum or semantic
  // answer table is used.
  bool has_mature_reafferent = false;
  for (std::uint32_t index = 0u; index < outcome_count; ++index) {
    const bool generic_component = antecedent_filter == 0u;
    const bool component_survives =
        outcome_reafferent_sources[index] != 0u &&
        (generic_component
             ? causal_relation_cross_context_component_survives_record_cuts(
                   state, relation, outcomes[index])
             : causal_relation_component_survives_record_cuts(
                   state, query_before, relation, outcomes[index]));
    if (component_survives)
      has_mature_reafferent = true;
  }
  const bool use_mature_reafferent =
      has_mature_reafferent &&
      (reafferent_bootstrap_source_free(state, trajectory, relation) ||
       // Once this live query contains the resident-generated bootstrap
       // action, its returned consequence is the only remaining unresolved
       // branch of that same physical episode.  A one-generated-tail query
       // therefore provides the chronology boundary that lets a mature
       // reafferent component supersede the ordinary bootstrap outcome;
       // before the action exists, the source-free check above still prevents
       // a returned component from pre-empting the first resident action.
       causal_relation_one_generated_tail(state, trajectory));
  for (std::uint32_t index = 0u; index < outcome_count; ++index) {
    if (outcome_sources[index] < 2u ||
        (use_mature_reafferent && outcome_reafferent_sources[index] == 0u) ||
        (!use_mature_reafferent && outcome_reafferent_sources[index] != 0u))
      continue;
    if (selected == kInvalid || outcome_sources[index] > selected_sources) {
      selected = outcomes[index];
      selected_sources = outcome_sources[index];
      selected_index = index;
      continue;
    }
    if (selected != outcomes[index] &&
        outcome_sources[index] == selected_sources) {
      // Equal independent support for different consequences is a genuine
      // live conflict. A strictly dominant closure may continue, but no
      // source order or physical slot may break a tie.
      result.ambiguous = true;
      return result;
    }
  }
  if (selected == kInvalid) return result;
  result.reafferent_sources = outcome_reafferent_sources[selected_index];
  result.ordinary_sources = outcome_ordinary_sources[selected_index];
  result.readiness_stage = 2u;

  std::uint64_t component_digest = 0u;
  std::uint64_t component_revision_digest = 0u;
  std::uint32_t source_revisions[kRelationCandidateScratchCapacity]{};
  std::uint32_t source_count = 0u;
  const std::uint32_t selected_outcome_index = selected_index;
  if (selected_outcome_index == kInvalid) return result;
  // A winning source pair contributes BOTH its antecedent (lane[1]==1,
  // lane[3]==before) and consequent (lane[1]==2, lane[3]==selected)
  // fragment to the digest. Filtering on lane[3] == selected alone only
  // ever matches the consequent half, so a mutated antecedent fragment
  // never changed component_revision_digest and re-derivation could not
  // detect it. Membership is by source (lane[4]) in the winning
  // source_keys/source_outcomes set instead, which both fragment kinds of
  // one pair share.
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    // The authority digest covers the complete selected source pair, not only
    // its consequence.  An antecedent lesion must invalidate the same permit
    // as a consequence or provenance lesion.
    if (!language_relation_source_eligible(state, record) ||
        record.lane[2] != relation ||
        record.lane[5] != 1u)
      continue;
    if (antecedent_filter != 0u) {
      bool exact_antecedent = false;
      for (std::uint32_t peer_slot = 0u; peer_slot < live_record_capacity(state);
           ++peer_slot) {
        const Record& peer = state->records[peer_slot];
        exact_antecedent |=
            language_relation_source_eligible(state, peer) &&
            peer.lane[4] == record.lane[4] && peer.lane[1] == 1u &&
            peer.lane[2] == relation && peer.lane[3] == antecedent_filter &&
            peer.lane[5] == 1u && peer.lane[6] == 0u;
      }
      if (!exact_antecedent) continue;
    }
    bool selected_source = false;
    for (std::uint32_t source = 0u; source < pair_count; ++source)
      selected_source |= source_outcomes[source] == selected &&
          source_keys[source] == record.lane[4];
    if (!selected_source) continue;
    component_digest ^= causal_relation_fragment_digest(record, 0u, false);
    component_revision_digest +=
        causal_relation_fragment_digest(record, 0u, true);
    bool source_seen = false;
    for (std::uint32_t index = 0u; index < source_count; ++index)
      source_seen |= source_revisions[index] == record.lane[4];
    if (!source_seen) source_revisions[source_count++] = record.lane[4];
  }
  if (source_count != selected_sources) return result;
  result.readiness_stage = 3u;

  // Every single fragment cut leaves another complete source pair. This is
  // the distributed closure authority for the generalized relation; no slot,
  // source order, or fixed quorum authorizes the output.
  for (std::uint32_t source = 0u; source < source_count; ++source) {
    std::uint32_t surviving = 0u;
    for (std::uint32_t other = 0u; other < source_count; ++other)
      surviving += source_revisions[other] != source ? 1u : 0u;
    if (surviving == 0u) return result;
  }
  result.readiness_stage = 4u;

  std::uint64_t external_provenance_digest = 0u;
  std::uint32_t external_leaves = 0u;
  for (std::uint32_t index = 0u; index < source_count; ++index) {
    std::uint64_t source_digest = 0u;
    std::uint32_t source_leaves = 0u;
    std::uint32_t source_failure = 0u;
    if (!causal_relation_external_source_provenance(
            state, source_revisions[index], &source_digest, &source_leaves,
            &source_failure)) {
      result.source_provenance_failure = source_failure;
      result.source_provenance_owner = source_revisions[index];
      return result;
    }
    external_provenance_digest ^= source_digest;
    external_leaves += source_leaves;
  }
  result.readiness_stage = 5u;
  std::uint64_t query_digest = 0u;
  std::uint32_t query_external_leaves = 0u;
  std::uint32_t query_leaves = 0u;
  if (!causal_relation_external_query_provenance(
          state, trajectory, &query_digest, &query_external_leaves,
          &query_leaves))
    return result;
  result.readiness_stage = 6u;
  result.source_provenance_digest = external_provenance_digest;
  result.source_external_leaves = external_leaves;
  result.query_context_digest = query_digest;
  result.query_context_leaves = query_leaves;
  external_provenance_digest ^= query_digest;
  external_leaves += query_external_leaves;

  const CausalRelationLiveWorldResolution live_world =
      causal_relation_live_world_value(state, source_revisions, source_count);
  if (live_world.bound && !live_world.valid) {
    result.applicable = true;
    result.query_linked = true;
    result.ready = false;
    return result;
  }
  result.readiness_stage = 7u;
  result.word = live_world.bound ? live_world.word : selected;
  result.extent = trajectory.lane[2];
  result.trajectory_owner = trajectory.lane[1];
  result.trajectory_revision = trajectory.revision;
  result.relation = relation;
  result.positive_sources = selected_sources;
  result.source_revision = source_revisions[0u];
  result.probe_steps = 1u;
  result.participating_records = source_count * 2u;
  result.independent_sources = source_count;
  result.source_contributions = source_count;
  result.maximum_source_contribution = 1u;
  result.contribution_concentration_q16 =
      0x10000u / source_count;
  result.singleton_supported_steps = 0u;
  result.minimum_probe_support = selected_sources;
  result.component_digest = component_digest;
  result.component_revision_digest = component_revision_digest;
  result.external_provenance_digest = external_provenance_digest;
  result.external_leaves = external_leaves;
  result.ready = selected_sources >= 2u && external_leaves != 0u &&
                 result.participating_records == result.source_contributions * 2u;
  if (result.ready) result.readiness_stage = 8u;
  return result;
}

// Prefer a closure grounded in the exact antecedent currently present on the
// query. A formed exact-context closure supersedes the generic relation
// ecology. Once exact-context participation exists, keep even an incomplete
// exact candidate visible so broader relation matter cannot veto the ordinary
// resident producer needed to acquire another independent returned source.
// Generic relation transfer remains available only while the current
// antecedent has no exact participation evidence at all. Readiness, ambiguity,
// cut closure, and provenance authority remain unchanged.
static BCC32_REWRITE_HD BCC32_RELATION_READER_NOINLINE
CausalRelationCandidate
collect_common_relation_candidate(const ResidentRewriteState* state,
                                  const Record& trajectory,
                                  std::uint32_t relation) {
  std::uint32_t antecedent = 0u;
  if (state != nullptr && trajectory_word_at(state, trajectory.lane[1], 0u,
                                             &antecedent) &&
      antecedent != 0u && antecedent != kInvalid) {
    const CausalRelationCandidate exact =
        collect_common_relation_candidate_filtered(
            state, trajectory, relation, antecedent);
    if (exact.ready || exact.ambiguous) return exact;
    if (exact.applicable) return exact;
    CausalRelationCandidate generic =
        collect_common_relation_candidate_filtered(state, trajectory, relation,
                                                   0u);
    if (generic.applicable) {
      // Do not discard an exact-context linkage merely because the broader
      // source scan is also applicable. The broader scan may lack a retained
      // source witness for this particular query while the exact resident
      // pair still proves that the relation is about the live cue. Preserve
      // that linkage on the broader candidate so a real multi-authority
      // conflict cannot become an unrelated generic fall-through. The
      // generic candidate still supplies the strongest available closure;
      // this boolean carries context identity only, never support.
      generic.query_linked = generic.query_linked || exact.query_linked;
      return generic;
    }
    return generic;
  }
  return collect_common_relation_candidate_filtered(state, trajectory, relation,
                                                    0u);
}

static BCC32_REWRITE_HD BCC32_RELATION_READER_NOINLINE
CausalRelationCandidate
collect_causal_relation_candidate(const ResidentRewriteState* state,
                                  const Record& trajectory) {
  CausalRelationCandidate result{};
  if (!causal_relation_query_trajectory(state, trajectory)) return result;
  std::uint32_t current = 0u;
  if (!trajectory_word_at(state, trajectory.lane[1], 0u, &current))
    return result;
  // A distributed bridge is a resident context edge, not a new singleton
  // relation. causal_relation_distributed_context_suffix only proves that a
  // real distributed context exists and that everything after it is
  // externally sourced; it does not know which of those external words is
  // the live relation connective, since unrelated raw contacts may have been
  // inserted between the bridge and the real connective (discourse must
  // survive intervening contacts, not just the immediately adjacent word).
  // Scan forward from the external suffix and let the same filtered common
  // reader used for the adjacent case decide relation applicability; the
  // first physically encountered applicable relation owns the probe, and an
  // incomplete/ambiguous relation abstains here rather than being skipped in
  // search of an easier later word -- that keeps the fail-closed behavior
  // every other relation candidate already has.
  std::uint32_t context_index = kInvalid;
  std::uint32_t external_suffix_begin = kInvalid;
  if (causal_relation_distributed_context_suffix(
          state, trajectory, &context_index, &external_suffix_begin)) {
    std::uint32_t context_word = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], context_index,
                            &context_word))
      return result;
    for (std::uint32_t index = external_suffix_begin;
         index < trajectory.lane[2]; ++index) {
      std::uint32_t relation = 0u;
      if (!trajectory_word_at(state, trajectory.lane[1], index, &relation))
        return result;
      const CausalRelationCandidate contextual =
          collect_common_relation_candidate_filtered(
              state, trajectory, relation, context_word, context_index);
      if (!contextual.applicable) continue;
      return contextual.query_linked ? contextual
                                     : CausalRelationCandidate{};
    }
    // A real distributed context existed, but no subsequent external word in
    // this trajectory engaged its relation ecology. Do not fall through into
    // the ordinary word-0 relation-chain reader below for this trajectory --
    // that reader's antecedent is word 0, which here is resident distributed
    // context, not a fresh external antecedent.
    return result;
  }
  // A two-word yielded query is the common-relation aperture: its antecedent
  // may be novel while its relation is already grounded by several external
  // source pairs. Resolve that distributed closure before enumerating exact
  // antecedent candidates, whose cut assay is needlessly scan-amplified for a
  // deliberately unseen cue. If no generic closure is applicable, retain
  // the exact pair reader as the stricter bootstrap fallback.
  if (trajectory.lane[2] == 2u ||
      causal_relation_one_generated_tail(state, trajectory)) {
    std::uint32_t relation = 0u;
    if (trajectory_word_at(state, trajectory.lane[1], 1u, &relation)) {
      const CausalRelationCandidate generalized =
          collect_common_relation_candidate(state, trajectory, relation);
      if (generalized.applicable &&
          (trajectory.lane[7] & kTrajectoryOpenInquiry) == 0u)
        // Keep a mature generalized candidate visible to the consensus
        // arbiter.  The arbiter already suppresses an unlinked generalized
        // candidate when an exact resident authority owns this query; hiding
        // it here also hides valid transfer to a novel antecedent.  An
        // open-inquiry continuation is different: its suspended Program and
        // action-return/reafferent authority must finish before a generic
        // relation can claim the same trajectory.
        return generalized;
    }
  }
  std::uint32_t weakest_sources = 0xffffffffu;
  std::uint32_t last_relation = 0u;
  std::uint32_t maximum_source = 0u;
  // A yielded relation query carries at most three probes. Keeping their raw
  // identities here lets the observer receipt be measured from the exact
  // participating source pairs after the complete chain has qualified.
  std::uint32_t probe_before[3]{};
  std::uint32_t probe_relation[3]{};
  std::uint32_t probe_after[3]{};
  std::uint32_t step_contribution_total = 0u;
  std::uint32_t singleton_supported_steps = 0u;
  for (std::uint32_t step = 1u; step < trajectory.lane[2]; ++step) {
    std::uint32_t relation = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], step, &relation))
      return CausalRelationCandidate{};
    std::uint32_t outcomes[kRelationCandidateScratchCapacity]{};
    std::uint32_t outcome_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& record = state->records[slot];
      if (!language_relation_source_eligible(state, record) ||
          record.lane[1] != 2u ||
          record.lane[2] != relation)
        continue;
      bool seen = false;
      for (std::uint32_t prior = 0u; prior < outcome_count; ++prior)
        seen |= outcomes[prior] == record.lane[3];
      if (!seen) {
        if (outcome_count == kRelationCandidateScratchCapacity)
          return result;
        outcomes[outcome_count++] = record.lane[3];
      }
    }
    std::uint32_t qualified = 0u;
    CausalRelationStepCandidate selected{};
    for (std::uint32_t index = 0u; index < outcome_count; ++index) {
      const CausalRelationStepCandidate candidate =
          collect_causal_relation_step_candidate(
              state, current, relation, outcomes[index]);
      if (candidate.positive_sources != 0u ||
          candidate.counter_sources != 0u)
        result.applicable = true;
      if (candidate.positive_sources != 0u ||
          candidate.counter_sources != 0u)
        result.query_linked = true;
      if (!candidate.cut_closed || candidate.counter_sources != 0u)
        continue;
      if (qualified == 0u || causal_relation_step_less(candidate, selected))
        selected = candidate;
      ++qualified;
    }
    if (qualified != 1u) {
      if (trajectory.lane[2] == 2u && step == 1u) {
        const CausalRelationCandidate generalized =
            collect_common_relation_candidate(state, trajectory, relation);
        if (generalized.applicable &&
            (trajectory.lane[7] & kTrajectoryOpenInquiry) == 0u)
          return generalized;
      }
      result.ambiguous = qualified > 1u;
      return result;
    }
    const std::uint32_t probe = step - 1u;
    probe_before[probe] = current;
    probe_relation[probe] = relation;
    probe_after[probe] = selected.after;
    step_contribution_total += selected.positive_sources;
    if (selected.positive_sources == 1u) ++singleton_supported_steps;
    current = selected.after;
    last_relation = relation;
    if (selected.positive_sources < weakest_sources)
      weakest_sources = selected.positive_sources;
    if (selected.maximum_source > maximum_source)
      maximum_source = selected.maximum_source;
  }

  // Attribute every accepted probe to the exact resident source pair that
  // supplied it. A source is counted only at its unique consequent fragment;
  // exact_causal_participation_pair rejects duplicate or extra fragments.
  // Thus these are measured contributions, not a proxy based on occupancy.
  std::uint32_t independent_sources = 0u;
  std::uint32_t source_contributions = 0u;
  std::uint32_t maximum_source_contribution = 0u;
  std::uint64_t component_digest = 0u;
  std::uint64_t component_revision_digest = 0u;
  std::uint32_t component_sources[kRelationCandidateScratchCapacity]{};
  std::uint32_t component_source_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (!language_relation_source_eligible(state, record) ||
        record.lane[5] != 1u)
      continue;
    std::uint32_t source_contribution = 0u;
    for (std::uint32_t probe = 0u; probe + 1u < trajectory.lane[2]; ++probe) {
      std::uint32_t consequent_slot = kInvalid;
      if (record.lane[2] != probe_relation[probe] ||
          !exact_causal_participation_pair(
              state, probe_before[probe], probe_relation[probe],
              probe_after[probe], record.lane[4], true, &consequent_slot))
        continue;
      const bool participating_fragment =
          (record.lane[1] == 1u &&
           record.lane[3] == probe_before[probe]) ||
          (record.lane[1] == 2u &&
           record.lane[3] == probe_after[probe]);
      if (!participating_fragment) continue;
      component_digest ^=
          causal_relation_fragment_digest(record, probe, false);
      component_revision_digest +=
          causal_relation_fragment_digest(record, probe, true);
      if (consequent_slot == slot) ++source_contribution;
    }
    if (source_contribution == 0u) continue;
    if (!source_already_counted(component_sources, component_source_count,
                                record.lane[4])) {
      if (component_source_count == kRelationCandidateScratchCapacity)
        return result;
      component_sources[component_source_count++] = record.lane[4];
    }
    ++independent_sources;
    source_contributions += source_contribution;
    if (source_contribution > maximum_source_contribution)
      maximum_source_contribution = source_contribution;
  }
  if (source_contributions == 0u ||
      source_contributions != step_contribution_total ||
      independent_sources == 0u ||
      component_source_count != independent_sources)
    return result;

  std::uint64_t external_provenance_digest = 0u;
  std::uint32_t external_leaves = 0u;
  for (std::uint32_t index = 0u; index < component_source_count; ++index) {
    std::uint64_t source_digest = 0u;
    std::uint32_t source_leaves = 0u;
    std::uint32_t source_failure = 0u;
    if (!causal_relation_external_source_provenance(
            state, component_sources[index], &source_digest,
            &source_leaves, &source_failure)) {
      result.source_provenance_failure = source_failure;
      result.source_provenance_owner = component_sources[index];
      return result;
    }
    external_provenance_digest ^= source_digest;
    external_leaves += source_leaves;
  }
  std::uint64_t query_digest = 0u;
  std::uint32_t query_external_leaves = 0u;
  std::uint32_t query_leaves = 0u;
  if (!causal_relation_external_query_provenance(
          state, trajectory, &query_digest, &query_external_leaves,
          &query_leaves))
    return result;
  result.source_provenance_digest = external_provenance_digest;
  result.source_external_leaves = external_leaves;
  result.query_context_digest = query_digest;
  result.query_context_leaves = query_leaves;
  external_provenance_digest ^= query_digest;
  external_leaves += query_external_leaves;

  result.word = current;
  result.extent = trajectory.lane[2];
  result.trajectory_owner = trajectory.lane[1];
  result.trajectory_revision = trajectory.revision;
  result.relation = last_relation;
  result.positive_sources = weakest_sources;
  result.source_revision = maximum_source;
  result.probe_steps = trajectory.lane[2] - 1u;
  result.participating_records = source_contributions * 2u;
  result.independent_sources = independent_sources;
  result.source_contributions = source_contributions;
  result.maximum_source_contribution = maximum_source_contribution;
  result.contribution_concentration_q16 = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(maximum_source_contribution) << 16u) /
      source_contributions);
  result.singleton_supported_steps = singleton_supported_steps;
  result.minimum_probe_support = weakest_sources;
  result.component_digest = component_digest;
  result.component_revision_digest = component_revision_digest;
  result.external_provenance_digest = external_provenance_digest;
  result.external_leaves = external_leaves;
  // Publication authority follows from the record-cut intervention already
  // applied to every selected step. The remaining checks close accounting;
  // source multiplicity and concentration stay observer receipts only.
  result.ready = weakest_sources != 0xffffffffu &&
                 result.probe_steps != 0u &&
                 result.source_contributions != 0u &&
                 result.external_leaves != 0u &&
                 result.participating_records ==
                     result.source_contributions * 2u;
  return result;
}

BCC32_REWRITE_HD inline void collect_resident_causal_relation_candidate(
    const ResidentRewriteState* state, const Record& trajectory,
    ProgramCandidateConsensus* consensus) {
  if (consensus == nullptr) return;
  const CausalRelationCandidate candidate =
      collect_causal_relation_candidate(state, trajectory);
  // A mature generalized relation can be residently real without belonging
  // to this exact antecedent.  It must not turn an unrelated component into a
  // conflict with an already selected exact Program/Span/VersionSpace route.
  // Once a relation has a cut-closed source fragment for this query, ordinary
  // multi-authority conflict and fail-closed behavior remains unchanged;
  // incomplete participation remains evidence until a later consequence can
  // close its own source component.
  if (consensus->have_candidate && !consensus->selected_from_constraint_relation &&
      !candidate.query_linked)
    return;
  if (candidate.ambiguous) {
    consensus->conflict = true;
    consensus->constraint_relation_ambiguous = true;
    return;
  }
  if (!candidate.ready) {
    // Applicable/query-linked participation is retained as resident evidence,
    // but until its own cut-closed consequence is ready it has not earned
    // authority to veto an independently grounded resident producer. Allow
    // that producer to keep generating the physical actions whose returned
    // consequences can grow the missing independent participation source.
    // Ambiguity remains fail-closed above, and a ready relation still enters
    // ordinary multi-authority conflict below.
    return;
  }
  // Participation is an independent resident authority class, not a Program
  // fallback. Any Program/Span/VersionSpace candidate already present makes
  // the combined continuation ambiguous even when its surface happens to
  // agree. Conversely the relation reader never calls the Program merge law.
  if (consensus->have_candidate || consensus->conflict) {
    consensus->conflict = true;
    return;
  }
  consensus->word = candidate.word;
  consensus->diagnostic_locus = kInvalid;
  consensus->extent = candidate.extent;
  consensus->have_candidate = true;
  consensus->saw_nonspan_at_selected_extent = true;
  consensus->selected_from_constraint_relation = true;
  consensus->constraint_relation_trajectory_owner = candidate.trajectory_owner;
  consensus->constraint_relation_trajectory_revision =
      candidate.trajectory_revision;
  consensus->constraint_relation = candidate.relation;
  consensus->constraint_relation_positive_sources = candidate.positive_sources;
  consensus->constraint_relation_source_revision = candidate.source_revision;
  consensus->constraint_relation_probe_steps = candidate.probe_steps;
  consensus->constraint_relation_participating_records =
      candidate.participating_records;
  consensus->constraint_relation_independent_sources =
      candidate.independent_sources;
  consensus->constraint_relation_source_contributions =
      candidate.source_contributions;
  consensus->constraint_relation_max_source_contribution =
      candidate.maximum_source_contribution;
  consensus->constraint_relation_contribution_concentration_q16 =
      candidate.contribution_concentration_q16;
  consensus->constraint_relation_singleton_supported_steps =
      candidate.singleton_supported_steps;
  consensus->constraint_relation_minimum_probe_support =
      candidate.minimum_probe_support;
  consensus->constraint_relation_component_digest =
      candidate.component_digest;
  consensus->constraint_relation_component_revision_digest =
      candidate.component_revision_digest;
  consensus->constraint_relation_source_provenance_digest =
      candidate.source_provenance_digest;
  consensus->constraint_relation_source_external_leaves =
      candidate.source_external_leaves;
  consensus->constraint_relation_query_context_digest =
      candidate.query_context_digest;
  consensus->constraint_relation_query_context_leaves =
      candidate.query_context_leaves;
  consensus->constraint_relation_external_provenance_digest =
      candidate.external_provenance_digest;
  consensus->constraint_relation_external_leaves = candidate.external_leaves;
  consensus->emission_permit.kind =
      ResidentEmissionPermitKind::kParticipationClosure;
  consensus->emission_permit.owner = candidate.trajectory_owner;
  consensus->emission_permit.participant_records =
      candidate.participating_records;
  // The permit's external-leaf field is source authority. The legacy
  // aggregate remains in the consensus telemetry fields above.
  consensus->emission_permit.external_leaves =
      candidate.source_external_leaves;
  consensus->emission_permit.independent_sources = candidate.independent_sources;
  consensus->emission_permit.source_contributions =
      candidate.source_contributions;
  consensus->emission_permit.source_external_leaves =
      candidate.source_external_leaves;
  consensus->emission_permit.query_context_leaves =
      candidate.query_context_leaves;
  consensus->emission_permit.topology_digest = candidate.component_digest;
  consensus->emission_permit.revision_digest =
      candidate.component_revision_digest;
  consensus->emission_permit.provenance_digest =
      candidate.source_provenance_digest;
  consensus->emission_permit.query_context_digest =
      candidate.query_context_digest;
}

BCC32_REWRITE_HD inline bool causal_relation_candidate_authoritative(
    const ResidentRewriteState* state,
    const ProgramCandidateConsensus& consensus) {
  if (state == nullptr || !consensus.selected_from_constraint_relation ||
      consensus.diagnostic_locus != kInvalid ||
      consensus.emission_permit.kind !=
          ResidentEmissionPermitKind::kParticipationClosure)
    return false;
  std::uint32_t trajectory_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& trajectory = state->records[slot];
    if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory ||
        trajectory.lane[1] !=
            consensus.constraint_relation_trajectory_owner ||
        trajectory.revision !=
            consensus.constraint_relation_trajectory_revision)
      continue;
    if (trajectory_slot != kInvalid) return false;
    trajectory_slot = slot;
  }
  if (trajectory_slot == kInvalid) return false;
  const CausalRelationCandidate current =
      collect_causal_relation_candidate(state, state->records[trajectory_slot]);
  return current.ready && !current.ambiguous &&
         current.independent_sources >= 2u &&
         current.source_contributions >= 2u &&
         current.word == consensus.word && current.extent == consensus.extent &&
         current.relation == consensus.constraint_relation &&
         current.positive_sources ==
             consensus.constraint_relation_positive_sources &&
         current.source_revision ==
             consensus.constraint_relation_source_revision &&
         current.probe_steps == consensus.constraint_relation_probe_steps &&
         current.participating_records ==
             consensus.constraint_relation_participating_records &&
         current.independent_sources ==
             consensus.constraint_relation_independent_sources &&
         current.source_contributions ==
             consensus.constraint_relation_source_contributions &&
         current.maximum_source_contribution ==
             consensus.constraint_relation_max_source_contribution &&
         current.contribution_concentration_q16 ==
             consensus.constraint_relation_contribution_concentration_q16 &&
         current.singleton_supported_steps ==
             consensus.constraint_relation_singleton_supported_steps &&
         current.minimum_probe_support ==
             consensus.constraint_relation_minimum_probe_support &&
         current.component_digest ==
             consensus.constraint_relation_component_digest &&
         current.component_revision_digest ==
             consensus.constraint_relation_component_revision_digest &&
         current.source_provenance_digest ==
             consensus.constraint_relation_source_provenance_digest &&
         current.source_external_leaves ==
             consensus.constraint_relation_source_external_leaves &&
         current.query_context_digest ==
             consensus.constraint_relation_query_context_digest &&
         current.query_context_leaves ==
             consensus.constraint_relation_query_context_leaves &&
         current.external_provenance_digest ==
             consensus.constraint_relation_external_provenance_digest &&
         current.external_leaves ==
             consensus.constraint_relation_external_leaves &&
         current.participating_records ==
             consensus.emission_permit.participant_records &&
         current.source_external_leaves ==
             consensus.emission_permit.external_leaves &&
         current.source_external_leaves ==
             consensus.emission_permit.source_external_leaves &&
         current.query_context_leaves ==
             consensus.emission_permit.query_context_leaves &&
         current.trajectory_owner == consensus.emission_permit.owner &&
         current.independent_sources ==
             consensus.emission_permit.independent_sources &&
         current.source_contributions ==
             consensus.emission_permit.source_contributions &&
         current.component_digest ==
             consensus.emission_permit.topology_digest &&
         current.component_revision_digest ==
             consensus.emission_permit.revision_digest &&
         current.source_provenance_digest ==
             consensus.emission_permit.provenance_digest &&
         current.query_context_digest ==
             consensus.emission_permit.query_context_digest;
}

BCC32_REWRITE_HD inline bool
resident_closure_emission_permit_authoritative(
    const ResidentRewriteState* state,
    const ProgramCandidateConsensus& consensus) {
  switch (consensus.emission_permit.kind) {
    case ResidentEmissionPermitKind::kParticipationClosure:
      return causal_relation_candidate_authoritative(state, consensus);
    case ResidentEmissionPermitKind::kNone:
    default:
      return false;
  }
}

#undef BCC32_RELATION_READER_NOINLINE
