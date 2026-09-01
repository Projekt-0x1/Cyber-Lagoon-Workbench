#pragma once

#include "bcc32_resident_open_inquiry.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_CLAUSE_DEPENDENCY_HD __host__ __device__
#else
#define BCC32_CLAUSE_DEPENDENCY_HD
#endif

// DISTRIBUTED PARTICIPATION REPLACEMENT (2026-08-14). The prior scaffold (see
// docs/diary) was a singleton witness -> singleton binding whose resolution
// was selected by a hand-authored one-bit parity classifier
// (target_index = clause_one.lane[4] & 1u) -- a localist-ledger /
// disguised-classifier pattern this project's doctrine rejects as a final
// capability. This file adds zero new authoritative Record forms beyond the
// clause-one witness (pure provenance, selects nothing by itself) and reuses
// the existing kFormConstraintParticipation substrate exactly as taught and
// read elsewhere in the resident causal-relation ecology.
//
// M8 intra-turn clause dependency: a second clause's resident content is
// causally constrained by the first clause's already-committed content
// rather than drawn from its own grounded evidence independently. Clause one
// is an ordinary deterministic grounded Program; once it fires, its raw
// generated word is witnessed into a persistent Record (multiple witnesses
// may coexist -- concurrent cognition is not bottlenecked by a global
// singleton). Clause two remains a genuine two-way grounded fork (the same
// exact-fork discovery already vetted by open_inquiry::collect_exact_fork).
// Which of its two live alternatives is dependent content is selected only
// by the same distributed causal-participation component-cut authority the
// generalized relation reader uses: at least two independently sourced
// (clause-one word, chosen clause-two word) episodes must remain cut-closed
// -- removable one fragment at a time while another complete source pair
// still carries the same transition. A single taught episode, a raw-word
// parity bit, or a semantic lookup can never select the output by itself.
//
// PRODUCTION WIRING STATUS: this module is a fully tested, self-contained
// capability (see the landing diary for all 13 covered controls: two-source
// redundancy, wrong-route, replay, lesion, source-withdrawal, single-record
// insufficiency, ambiguity, inverse/expiry, stale rederivation, multi-pair
// coexistence). It is NOT yet wired into advance_resident_program_once's
// production dispatcher: an earlier attempt to forward-declare and call its
// bridge functions from causal_rewrite_program_execution.inl (a file
// processed by every translation unit in this codebase) proved fragile --
// depending on which file first triggers causal_rewrite_universe.cuh's
// header chain, pulling in this module's own dependency on
// bcc32_resident_open_inquiry.cuh from that universally-shared file
// intermittently broke unrelated targets (observed: bcc32_cuda_resident_
// causal_relation_reader_contract once, bcc32_cuda_resident_pending_means_
// contract once) via circular-include ordering. Safely wiring this in needs
// either a verified weak-symbol default (untested here) or an explicit
// opt-in call site at the real production ingestion point, not a change to
// the universally-compiled dispatcher; see the landing diary's next_rung.
namespace substrate::bcc32::causal_rewrite::clause_dependency {

using causal_rewrite::allocate_record;
using causal_rewrite::apply_physical_lesion;
using causal_rewrite::CausalRelationStepCandidate;
using causal_rewrite::causal_product_has_live_counterevidence;
using causal_rewrite::collect_causal_relation_step_candidate;
using causal_rewrite::free_record_count;
using causal_rewrite::kFormConstraintParticipation;
using causal_rewrite::kFormProgram;
using causal_rewrite::kFormTrajectory;
using causal_rewrite::kInvalid;
using causal_rewrite::kRecordCapacity;
using causal_rewrite::live_record_capacity;
using causal_rewrite::make_record_owner;
using causal_rewrite::Record;
using causal_rewrite::refresh_receipt;
using causal_rewrite::ResidentRewriteState;
using causal_rewrite::version_space_program_grounded;

inline constexpr std::uint32_t kFormClauseDependencyClauseOne = 0x1f6b8ad4u;
inline constexpr std::uint32_t kClauseOneWitnessed = 1u;

// Fixed architectural channel identifier for M8 intra-turn clause-dependency
// participation. Structurally this is an ordinary lane[2] "relation" word,
// exactly like any relation the generalized reader learns from raw external
// contact; it is reserved by this file rather than supplied through contact.
// It is not a semantic answer, lookup key, or per-episode hash: it selects
// nothing by itself. The winning clause-two alternative is determined only
// by the distributed, independently-sourced participation component below.
inline constexpr std::uint32_t kClauseDependencyRelation = 0x4ea1e001u;

// Clause-one witness lanes: [1] owner; [2] producing grounded Program owner
// (also the participation source identity used below -- content-addressed,
// so replaying the same grounded Program yields the same source rather than
// fabricating new independent support); [3] that Program's revision at
// capture time; [4] the raw generated word; [7] flags.
BCC32_CLAUSE_DEPENDENCY_HD inline bool capture_clause_one_from_last_generated(
    ResidentRewriteState* state, std::uint32_t* producer_owner_out = nullptr,
    std::uint32_t* producer_revision_out = nullptr,
    std::uint32_t* word_out = nullptr,
    std::uint32_t* clause_one_slot_out = nullptr) {
  if (producer_owner_out != nullptr) *producer_owner_out = kInvalid;
  if (producer_revision_out != nullptr) *producer_revision_out = 0u;
  if (word_out != nullptr) *word_out = kInvalid;
  if (clause_one_slot_out != nullptr) *clause_one_slot_out = kInvalid;
  if (state == nullptr || state->fault != 0u ||
      state->generated_word_valid == 0u)
    return false;
  const std::uint32_t producer_slot = state->generated_locus;
  if (producer_slot >= live_record_capacity(state)) return false;
  const Record& producer = state->records[producer_slot];
  if (producer.matter_q8 == 0u || producer.lane[0] != kFormProgram ||
      !version_space_program_grounded(state, producer_slot, true) ||
      causal_product_has_live_counterevidence(state, producer_slot))
    return false;
  if (free_record_count(state) < 1u) return false;
  const std::uint32_t owner = make_record_owner(
      state, causal_rewrite::rewrite_mix(kFormClauseDependencyClauseOne,
                                         producer.lane[1],
                                         state->generated_word));
  if (owner == kInvalid) return false;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& clause_one = state->records[slot];
  clause_one.lane[0] = kFormClauseDependencyClauseOne;
  clause_one.lane[1] = owner;
  clause_one.lane[2] = producer.lane[1];
  clause_one.lane[3] = producer.revision;
  clause_one.lane[4] = state->generated_word;
  clause_one.lane[7] = kClauseOneWitnessed;
  ++clause_one.revision;
  ++state->revision;
  refresh_receipt(state);
  if (producer_owner_out != nullptr) *producer_owner_out = producer.lane[1];
  if (producer_revision_out != nullptr)
    *producer_revision_out = producer.revision;
  if (word_out != nullptr) *word_out = state->generated_word;
  if (clause_one_slot_out != nullptr) *clause_one_slot_out = slot;
  return true;
}

// Liveness of an exact caller-held witness identity (producer owner and
// revision), never an implicit global search. A physical lesion of that one
// witness Record breaks only dependencies keyed on it; unrelated concurrent
// witnesses are untouched.
BCC32_CLAUSE_DEPENDENCY_HD inline bool clause_one_witness_live(
    const ResidentRewriteState* state, std::uint32_t producer_owner,
    std::uint32_t producer_revision, std::uint32_t* word_out = nullptr) {
  if (word_out != nullptr) *word_out = kInvalid;
  if (state == nullptr || producer_owner == 0u || producer_owner == kInvalid)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormClauseDependencyClauseOne ||
        record.lane[2] != producer_owner)
      continue;
    if (record.lane[3] != producer_revision ||
        (record.lane[7] & kClauseOneWitnessed) == 0u)
      return false;
    if (word_out != nullptr) *word_out = record.lane[4];
    return true;
  }
  return false;
}

// Whether a participation fragment of this exact shape already exists.
// Teaching the same source repeatedly must stay recurrent, never fabricate
// additional independent support -- this is the sole guard for that.
BCC32_CLAUSE_DEPENDENCY_HD inline bool clause_dependency_fragment_exists(
    const ResidentRewriteState* state, std::uint32_t fragment_kind,
    std::uint32_t endpoint, std::uint32_t source_revision) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormConstraintParticipation ||
        record.lane[1] != fragment_kind ||
        record.lane[2] != kClauseDependencyRelation ||
        record.lane[3] != endpoint || record.lane[4] != source_revision)
      continue;
    return true;
  }
  return false;
}

// Admit one M8 teaching episode as ordinary kFormConstraintParticipation
// matter, in exactly the physical shape causal_participation_record() (the
// generalized relation reader's own validity check) already requires: one
// antecedent fragment (lane[1]==1) and one consequent fragment (lane[1]==2),
// never a single Record holding both endpoints. Source identity is the
// producing grounded Program's own owner: independently grounded Programs
// are independent sources; replaying one Program's firing is recurrent,
// never new support.
BCC32_CLAUSE_DEPENDENCY_HD inline bool teach_clause_dependency_participation(
    ResidentRewriteState* state, std::uint32_t producer_owner,
    std::uint32_t clause_one_word, std::uint32_t chosen_clause_two_word) {
  if (state == nullptr || producer_owner == 0u || producer_owner == kInvalid ||
      clause_one_word == 0u || clause_one_word == kInvalid ||
      chosen_clause_two_word == 0u || chosen_clause_two_word == kInvalid)
    return false;
  const bool have_antecedent = clause_dependency_fragment_exists(
      state, 1u, clause_one_word, producer_owner);
  const bool have_consequent = clause_dependency_fragment_exists(
      state, 2u, chosen_clause_two_word, producer_owner);
  const std::uint32_t needed =
      (have_antecedent ? 0u : 1u) + (have_consequent ? 0u : 1u);
  if (needed > free_record_count(state)) return false;
  if (!have_antecedent) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& record = state->records[slot];
    record.lane[0] = kFormConstraintParticipation;
    record.lane[1] = 1u;
    record.lane[2] = kClauseDependencyRelation;
    record.lane[3] = clause_one_word;
    record.lane[4] = producer_owner;
    record.lane[5] = 1u;
    record.lane[6] = 0u;
    ++record.revision;
    ++state->revision;
  }
  if (!have_consequent) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& record = state->records[slot];
    record.lane[0] = kFormConstraintParticipation;
    record.lane[1] = 2u;
    record.lane[2] = kClauseDependencyRelation;
    record.lane[3] = chosen_clause_two_word;
    record.lane[4] = producer_owner;
    record.lane[5] = 1u;
    record.lane[6] = 0u;
    ++record.revision;
    ++state->revision;
  }
  refresh_receipt(state);
  return true;
}

struct ClauseDependencyCandidate {
  std::uint32_t word = kInvalid;
  std::uint32_t clause_one_word = kInvalid;
  bool applicable = false;
  bool ready = false;
  bool ambiguous = false;
};

// Authority is the same component cut-closure the generalized relation
// reader applies: every participating fragment of the winning transition
// must be removable while another complete independently sourced pair still
// carries the same clause-one -> clause-two transition. Exactly one live
// fork alternative may qualify; zero or two both abstain.
BCC32_CLAUSE_DEPENDENCY_HD inline ClauseDependencyCandidate
collect_clause_dependency_candidate(const ResidentRewriteState* state,
                                    std::uint32_t clause_one_word,
                                    const Record& clause_two_trajectory) {
  ClauseDependencyCandidate result{};
  result.clause_one_word = clause_one_word;
  if (state == nullptr || clause_one_word == 0u || clause_one_word == kInvalid)
    return result;
  open_inquiry::AlternativeSnapshot alternatives[2]{};
  if (!open_inquiry::collect_exact_fork(state, clause_two_trajectory,
                                        alternatives) ||
      alternatives[0].consequence == alternatives[1].consequence)
    return result;
  std::uint32_t qualified = 0u;
  std::uint32_t winner = kInvalid;
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const CausalRelationStepCandidate step = collect_causal_relation_step_candidate(
        state, clause_one_word, kClauseDependencyRelation,
        alternatives[index].consequence);
    if (step.positive_sources != 0u || step.counter_sources != 0u)
      result.applicable = true;
    if (!step.cut_closed || step.counter_sources != 0u) continue;
    winner = alternatives[index].consequence;
    ++qualified;
  }
  if (qualified != 1u) {
    result.ambiguous = qualified > 1u;
    return result;
  }
  result.word = winner;
  result.ready = true;
  return result;
}

// Re-derive the candidate fresh from live resident matter and compare
// against a caller's earlier claim. This is the same "rederive independently
// before the emitter commits" pattern the relation reader uses: an
// intervening lesion, source withdrawal, or defeating contradiction changes
// live participation matter without touching this comparison's inputs, so a
// stale claim is rejected here rather than trusted from an earlier resolve.
BCC32_CLAUSE_DEPENDENCY_HD inline bool clause_dependency_candidate_authoritative(
    const ResidentRewriteState* state, std::uint32_t clause_one_word,
    const Record& clause_two_trajectory, std::uint32_t claimed_word) {
  const ClauseDependencyCandidate fresh = collect_clause_dependency_candidate(
      state, clause_one_word, clause_two_trajectory);
  return fresh.ready && !fresh.ambiguous && fresh.word == claimed_word;
}

// Resolve clause two's dependent content end to end for a caller-held clause
// one identity. This performs no Record allocation and owns no output
// register; a qualified result is read-only distributed-participation
// evidence for the caller to enter its own emission/consensus path.
BCC32_CLAUSE_DEPENDENCY_HD inline bool resolve_clause_two_dependency(
    const ResidentRewriteState* state, std::uint32_t producer_owner,
    std::uint32_t producer_revision, std::uint32_t clause_two_trajectory_slot,
    std::uint32_t* resolved_out) {
  if (resolved_out != nullptr) *resolved_out = kInvalid;
  if (state == nullptr || clause_two_trajectory_slot >= live_record_capacity(state))
    return false;
  std::uint32_t clause_one_word = kInvalid;
  if (!clause_one_witness_live(state, producer_owner, producer_revision,
                               &clause_one_word))
    return false;
  const Record& trajectory = state->records[clause_two_trajectory_slot];
  if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory)
    return false;
  const ClauseDependencyCandidate candidate =
      collect_clause_dependency_candidate(state, clause_one_word, trajectory);
  if (!candidate.ready || candidate.ambiguous) return false;
  if (resolved_out != nullptr) *resolved_out = candidate.word;
  return true;
}

// Clause two's unattached baseline: its own numerically lower live grounded
// alternative, discovered the same vetted way, with no dependency consulted
// at all. This is what an unrelated or absent clause one leaves behind.
BCC32_CLAUSE_DEPENDENCY_HD inline bool resolve_clause_two_independent(
    const ResidentRewriteState* state, std::uint32_t clause_two_trajectory_slot,
    std::uint32_t* resolved_out) {
  if (resolved_out != nullptr) *resolved_out = kInvalid;
  if (state == nullptr || clause_two_trajectory_slot >= live_record_capacity(state))
    return false;
  const Record& trajectory = state->records[clause_two_trajectory_slot];
  if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory)
    return false;
  open_inquiry::AlternativeSnapshot alternatives[2]{};
  if (!open_inquiry::collect_exact_fork(state, trajectory, alternatives))
    return false;
  if (resolved_out != nullptr) *resolved_out = alternatives[0].consequence;
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite::clause_dependency

#undef BCC32_CLAUSE_DEPENDENCY_HD
