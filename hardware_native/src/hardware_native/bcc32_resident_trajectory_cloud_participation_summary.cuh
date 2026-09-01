#pragma once

#include "bcc32_resident_page_constraint_iterator.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_TRAJECTORY_CLOUD_SUMMARY_HD __host__ __device__
#else
#define BCC32_TRAJECTORY_CLOUD_SUMMARY_HD
#endif

// RWR0 trajectory-level constraint-cloud participation summary.
//
// A downstream consumer of the RWR0 chain -- the dialogue adapter marked
// READY this session (SHA 673961c9c3, "depends on cloud participation
// header"), or a future ticketed-revision adapter -- needs more than one raw
// word's participation flag: it needs to know, over a whole resident
// trajectory, how many terms currently ground in the constraint cloud and
// where the first one sits, so it can decide (elsewhere, not here) whether
// that trajectory is a candidate for inquiry, a ticketed action, or a
// revision replay. This header adds exactly one such accessor.
//
// This is a pure fold over resident_page_constraint_iterator::
// page_constraint_term_at, called once per trajectory-relative index, exactly
// the same per-term resolution the predecessor iterator already established
// and already contract-proved page-crossing for. Nothing here re-derives
// participation; nothing here re-derives page resolution. The loop below is
// structurally the same forward walk as
// resident_page_constraint_iterator::walk_page_constraint_terms, minus the
// caller-supplied output array (a summary has no per-term output to buffer),
// so it carries no output-buffer bound at all and therefore cannot be
// mistaken for a policy cap:
//   * no new Record form is introduced;
//   * no Record is written, allocated, or cleared by anything in this file;
//   * no new authority, cache, or projection is created -- total_terms,
//     participating_terms, and first_participating_index are recomputed by
//     walking the live records[] population (transitively, through
//     page_constraint_term_at) on every call, never stored;
//   * no fixed page/event cap is reintroduced -- the fold below stops only
//     when page_constraint_term_at reports an unresolved index (the real
//     physical end of the admitted trajectory, however many pages that
//     spans), which is itself bounded only by trajectory_word_at's existing
//     kRecordCapacity physical scan bound, exactly as
//     walk_page_constraint_terms already documents;
//   * no numeric class label or majority vote is computed -- the three
//     outputs are a plain count of resolved terms, a plain count of the
//     subset that flagged participates != 0, and the lowest index among
//     that subset (kInvalid when the subset is empty). None of the three is
//     compared, thresholded, or converted into a verdict here; that decision
//     belongs entirely to whatever caller consumes these raw counts.
namespace substrate::bcc32::resident_trajectory_cloud_participation_summary {

namespace rewrite = substrate::bcc32::causal_rewrite;
namespace page_constraint = substrate::bcc32::resident_page_constraint_iterator;

// Folds page_constraint_term_at over every page-resolvable term of the
// trajectory owned by `owner`, starting at index 0, and reports three plain
// aggregate facts:
//   * out_total_terms: how many trajectory-relative indices resolved at all
//     (the real, page-continued trajectory length);
//   * out_participating_terms: how many of those resolved terms currently
//     flag participates != 0 (word_participates_in_constraint_cloud true);
//   * out_first_participating_index: the lowest trajectory-relative index
//     among the participating subset, or rewrite::kInvalid if that subset is
//     empty.
// Any output pointer may be nullptr if the caller does not need that field.
// Returns false (leaving every requested output at its zeroed/kInvalid
// default) only when `state` is null or `owner` resolves no trajectory at
// all (out_total_terms stays 0u in that case too, which is itself a
// legitimate, distinguishable summary of an empty/absent trajectory).
BCC32_TRAJECTORY_CLOUD_SUMMARY_HD inline bool
trajectory_cloud_participation_summary(const rewrite::ResidentRewriteState* state,
                                        std::uint32_t owner,
                                        std::uint32_t* out_total_terms,
                                        std::uint32_t* out_participating_terms,
                                        std::uint32_t* out_first_participating_index) {
  if (out_total_terms != nullptr) *out_total_terms = 0u;
  if (out_participating_terms != nullptr) *out_participating_terms = 0u;
  if (out_first_participating_index != nullptr)
    *out_first_participating_index = rewrite::kInvalid;
  if (state == nullptr) return false;

  std::uint32_t total_terms = 0u;
  std::uint32_t participating_terms = 0u;
  std::uint32_t first_participating_index = rewrite::kInvalid;
  std::uint32_t index = 0u;
  while (true) {
    const page_constraint::PageConstraintTerm term =
        page_constraint::page_constraint_term_at(state, owner, index);
    if (term.resolved == 0u) break;
    ++total_terms;
    if (term.participates != 0u) {
      ++participating_terms;
      if (first_participating_index == rewrite::kInvalid)
        first_participating_index = index;
    }
    ++index;
  }

  if (out_total_terms != nullptr) *out_total_terms = total_terms;
  if (out_participating_terms != nullptr)
    *out_participating_terms = participating_terms;
  if (out_first_participating_index != nullptr)
    *out_first_participating_index = first_participating_index;
  return true;
}

}  // namespace substrate::bcc32::resident_trajectory_cloud_participation_summary
