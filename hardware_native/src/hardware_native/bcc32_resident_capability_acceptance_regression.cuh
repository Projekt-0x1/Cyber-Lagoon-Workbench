#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_ACCEPT_REGR_HD __host__ __device__
#else
#define BCC32_ACCEPT_REGR_HD
#endif

// Hostile acceptance regression harness: five generic, read-only detectors
// for the five founder-named capability-acceptance anti-patterns. This is
// meta-tooling, not a production capability -- it never allocates against a
// live resident universe and never writes a Record. It takes a caller-built
// synthetic ResidentRewriteState (constructed by a contract test the same
// way every other contract in this codebase builds fixtures: allocate_record
// plus explicit lane assignment) and answers, per named anti-pattern,
// whether the supplied fixture exhibits the shortcut.
//
// The Record forms below are local to this harness. They do not extend or
// alias any production form; they exist only to let a contract test express
// each anti-pattern's minimal falsifiable shape using the established
// Record-slot idiom (kInvalid, allocate_record, a single-pass uniqueness
// scan over kRecordCapacity -- see bcc32_resident_open_inquiry.cuh's
// unique_active_inquiry / unique_owned_slot for the same shape).
namespace substrate::bcc32::causal_rewrite::capability_acceptance_regression {

// --- Anti-pattern 1: stored-trajectory replay -----------------------------
// TrainingTrajectory header: lane[1] owner; lane[2] word count.
// TrainingTrajectoryTerm:     lane[1] owner; lane[2] index; lane[3] word.
inline constexpr std::uint32_t kFormRegressionTrainingTrajectory = 0xac100001u;
inline constexpr std::uint32_t kFormRegressionTrainingTrajectoryTerm = 0xac100002u;
// ClaimedGeneration header: lane[1] owner; lane[2] word count (a "held-out
// generation" the capability under test is claiming as its own output).
// ClaimedGenerationTerm:     lane[1] owner; lane[2] index; lane[3] word.
inline constexpr std::uint32_t kFormRegressionClaimedGeneration = 0xac100003u;
inline constexpr std::uint32_t kFormRegressionClaimedGenerationTerm = 0xac100004u;

// --- Anti-pattern 2: episode remainder presented as an answer -------------
// Question header: lane[1] owner (no length needed by the detector).
// QuestionTerm:      lane[1] owner; lane[2] index; lane[3] word.
inline constexpr std::uint32_t kFormRegressionQuestion = 0xac100005u;
inline constexpr std::uint32_t kFormRegressionQuestionTerm = 0xac100006u;
// Reply: lane[1] owner; lane[2] claimed binding owner (kInvalid if the
// reply carries no causal reference to any question content at all --
// exactly "leftover post-episode resident state"); lane[4] the reply's own
// content word, which must match the binding's bound word to verify.
inline constexpr std::uint32_t kFormRegressionReply = 0xac100007u;
// ReplyBinding: lane[1] owner; lane[2] question owner it claims to derive
// from; lane[3] index into that question's own term sequence; lane[4] the
// word claimed, at that index, to have driven the reply. The detector
// re-derives the question word at that index independently and only
// accepts the link if it is exact -- a fabricated or dangling binding is
// treated exactly like no binding at all.
inline constexpr std::uint32_t kFormRegressionReplyBinding = 0xac100008u;

// --- Anti-pattern 3: host-expected-output oracle acceptance ---------------
// OracleCheck: lane[1] owner; lane[2] owner of a caller-supplied fixed
// expected-value record actually consulted by acceptance (kInvalid if
// acceptance never referenced one); lane[3] owner of a resident Record
// state actually consulted by acceptance (kInvalid if none). A majority
// vote or numeric label tallied from caller-provided examples is the same
// anti-pattern under this model: whatever record supplies the accept/
// reject threshold is the "caller-supplied expected value," so its owner
// belongs in lane[2] regardless of how many hops of arithmetic produced it.
inline constexpr std::uint32_t kFormRegressionOracleCheck = 0xac100009u;

// --- Anti-pattern 4: unpaid associative capacity ---------------------------
// Association: lane[1] owner; lane[2] claimed support/capacity (0 = claims
// nothing, so there is nothing to check).
inline constexpr std::uint32_t kFormRegressionAssociation = 0xac10000au;
// MatterDebit: lane[1] owner (must match the Association's owner); lane[2]
// the matter_q8-denominated amount actually debited from the resident
// ledger to fund that claim. A real debit must reach at least one whole
// kRecordMatterQ8 unit -- the same unit every ordinary Record is priced in.
inline constexpr std::uint32_t kFormRegressionMatterDebit = 0xac10000bu;

// --- Anti-pattern 5: singleton-only relation support -----------------------
// RelationEpisode: lane[1] relation owner; lane[2] episode id. One record
// per (relation, episode) observation. A relation supported by only one
// distinct episode id, however many redundant records restate it, is
// unsupported by reuse.
inline constexpr std::uint32_t kFormRegressionRelationEpisode = 0xac10000cu;

// A single-pass, fail-closed lookup for exactly one Record of a given form
// owned by a given owner. Ambiguous (more than one match) resolves to
// kInvalid, matching the rest of this codebase's uniqueness-scan idiom.
BCC32_ACCEPT_REGR_HD inline std::uint32_t regression_unique_owned(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != form ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

// A single-pass, fail-closed lookup for the word at (term_form, owner,
// index). Ambiguous or missing resolves to false.
BCC32_ACCEPT_REGR_HD inline bool regression_term_word_at(
    const ResidentRewriteState* state, std::uint32_t term_form,
    std::uint32_t owner, std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr) return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != term_form ||
        record.lane[1] != owner || record.lane[2] != index)
      continue;
    if (found != kInvalid) return false;
    found = slot;
  }
  if (found == kInvalid) return false;
  *word = state->records[found].lane[3];
  return true;
}

// Detector 1: flags a claimed held-out generation that is byte-identical,
// word for word over its full length, to some resident training
// trajectory. A genuinely different generation (different length, or any
// differing word) is not flagged.
BCC32_ACCEPT_REGR_HD inline bool detect_stored_trajectory_replay(
    const ResidentRewriteState* state,
    std::uint32_t claimed_generation_owner) {
  if (state == nullptr) return false;
  const std::uint32_t claimed_slot = regression_unique_owned(
      state, kFormRegressionClaimedGeneration, claimed_generation_owner);
  if (claimed_slot == kInvalid) return false;
  const std::uint32_t claimed_length = state->records[claimed_slot].lane[2];
  if (claimed_length == 0u) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& training = state->records[slot];
    if (training.matter_q8 == 0u ||
        training.lane[0] != kFormRegressionTrainingTrajectory ||
        training.lane[2] != claimed_length)
      continue;
    bool identical = true;
    for (std::uint32_t index = 0u; index < claimed_length; ++index) {
      std::uint32_t claimed_word = 0u;
      std::uint32_t training_word = 0u;
      if (!regression_term_word_at(state,
                                   kFormRegressionClaimedGenerationTerm,
                                   claimed_generation_owner, index,
                                   &claimed_word) ||
          !regression_term_word_at(state,
                                   kFormRegressionTrainingTrajectoryTerm,
                                   training.lane[1], index,
                                   &training_word) ||
          claimed_word != training_word) {
        identical = false;
        break;
      }
    }
    if (identical) return true;
  }
  return false;
}

// Detector 2: flags a reply with no verifiable causal link to the actual
// question content it claims to answer -- either no binding at all
// (leftover post-episode resident state emitted as if it were an answer),
// a binding that does not resolve, or a binding whose claimed source word
// does not match the question's own independently stored content (a
// fabricated link is exactly as unsupported as no link). A reply that
// verifiably traces to the question's own content is not flagged.
BCC32_ACCEPT_REGR_HD inline bool detect_episode_remainder_as_answer(
    const ResidentRewriteState* state, std::uint32_t reply_owner) {
  if (state == nullptr) return false;
  const std::uint32_t reply_slot =
      regression_unique_owned(state, kFormRegressionReply, reply_owner);
  if (reply_slot == kInvalid) return false;
  const Record& reply = state->records[reply_slot];
  const std::uint32_t binding_owner = reply.lane[2];
  if (binding_owner == kInvalid) return true;
  const std::uint32_t binding_slot = regression_unique_owned(
      state, kFormRegressionReplyBinding, binding_owner);
  if (binding_slot == kInvalid) return true;
  const Record& binding = state->records[binding_slot];
  const std::uint32_t question_owner = binding.lane[2];
  const std::uint32_t question_index = binding.lane[3];
  const std::uint32_t bound_word = binding.lane[4];
  std::uint32_t question_word = 0u;
  if (!regression_term_word_at(state, kFormRegressionQuestionTerm,
                               question_owner, question_index,
                               &question_word) ||
      question_word != bound_word || reply.lane[4] != bound_word)
    return true;
  return false;
}

// Detector 3: flags acceptance logic that consults a caller-supplied fixed
// expected-value record at all (the exact host-expected-output/oracle
// shortcut, including a disguised classifier such as a caller-numeric
// label or a majority vote tallied over caller-provided examples --
// whatever supplies the accept/reject threshold is the caller-supplied
// expected value). Acceptance derived purely from resident Record state,
// with no caller-supplied expected value consulted, is not flagged.
BCC32_ACCEPT_REGR_HD inline bool detect_host_expected_output_oracle(
    const ResidentRewriteState* state, std::uint32_t check_owner) {
  if (state == nullptr) return false;
  const std::uint32_t slot = regression_unique_owned(
      state, kFormRegressionOracleCheck, check_owner);
  if (slot == kInvalid) return false;
  return state->records[slot].lane[2] != kInvalid;
}

// Detector 4: flags an association/relation that claims support or
// capacity (a nonzero claim) without a matching matter/energy debit of at
// least one real accounted unit in the resident ledger. An association
// backed by a real debit of at least kRecordMatterQ8 is not flagged.
BCC32_ACCEPT_REGR_HD inline bool detect_unpaid_association_capacity(
    const ResidentRewriteState* state, std::uint32_t association_owner) {
  if (state == nullptr) return false;
  const std::uint32_t assoc_slot = regression_unique_owned(
      state, kFormRegressionAssociation, association_owner);
  if (assoc_slot == kInvalid) return false;
  if (state->records[assoc_slot].lane[2] == 0u) return false;
  const std::uint32_t debit_slot = regression_unique_owned(
      state, kFormRegressionMatterDebit, association_owner);
  if (debit_slot == kInvalid) return true;
  return state->records[debit_slot].lane[2] < kRecordMatterQ8;
}

// Detector 5: flags a relation whose probed support resolves to exactly
// one distinct episode id -- a lucky singleton lookup, not reusable
// support -- however many redundant records restate that same episode. A
// relation whose support spans two or more distinct episode ids is not
// flagged.
BCC32_ACCEPT_REGR_HD inline bool detect_singleton_only_relation_support(
    const ResidentRewriteState* state, std::uint32_t relation_owner) {
  if (state == nullptr) return false;
  bool found_any = false;
  bool multiple_episodes = false;
  std::uint32_t reference_episode = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormRegressionRelationEpisode ||
        record.lane[1] != relation_owner)
      continue;
    if (!found_any) {
      found_any = true;
      reference_episode = record.lane[2];
    } else if (record.lane[2] != reference_episode) {
      multiple_episodes = true;
    }
  }
  return found_any && !multiple_episodes;
}

}  // namespace substrate::bcc32::causal_rewrite::capability_acceptance_regression
