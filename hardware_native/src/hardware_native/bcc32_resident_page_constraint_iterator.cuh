#pragma once

#include "causal_rewrite_universe.cuh"
#include "bcc32_resident_causal_constraint_participation.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_PAGE_CONSTRAINT_HD __host__ __device__
#else
#define BCC32_PAGE_CONSTRAINT_HD
#endif

// RWR0 page-aware + constraint-cloud-aware term accessor. Every consumer that
// walks a resident trajectory beyond the single-page assumption must resolve
// terms through the real, unbounded page chain (kFormTrajectoryPage /
// trajectory_word_at / kRecordCapacity as a physical scan bound only -- there
// is no reintroduced fixed page or event ceiling here). This header adds one
// centralized, reusable, read-only accessor that composes that page walk with
// a second re-derivation over the already-landed constraint-participation
// Record population (kFormConstraintParticipation), so a caller can learn, in
// one call, both a page-continued raw word and whether that exact word
// currently participates in the resident constraint cloud.
//
// This is pure derivation over existing Records:
//   * no new Record form is introduced;
//   * no Record is written, allocated, or cleared by anything in this file;
//   * no new authority, cache, or projection is created -- every answer is
//     recomputed from the live records[] population on every call;
//   * no fixed page/event cap is reintroduced (max_terms below is only an
//     output-buffer bound the caller supplies, exactly like every other
//     bounded-output accessor in this codebase; it is not consulted by the
//     underlying page walk, which stops only at the first unresolved index);
//   * no numeric class label or majority vote is used to decide participation
//     -- word_participates_in_constraint_cloud is a plain existential scan
//     for one exact, structurally valid fragment carrying the exact raw word
//     value, matching is_participation() as already defined in the landed
//     participation header.
namespace substrate::bcc32::resident_page_constraint_iterator {

namespace rewrite = substrate::bcc32::causal_rewrite;
namespace participation = substrate::bcc32::resident_causal_constraint_participation;

// One page-resolved trajectory term, with its constraint-cloud participation
// flagged alongside it. The result deliberately contains no Record slot or
// owner: a physical address would turn an observer readout into a privileged
// localist handle instead of a re-derived property of the full live cloud.
struct PageConstraintTerm {
  std::uint32_t word = 0u;
  std::uint32_t resolved = 0u;
  std::uint32_t participates = 0u;
};

// Does a live, structurally valid constraint-participation fragment currently
// carry exactly this raw word, either as its relation atom (lane[2]) or as
// its endpoint atom (lane[3])? Both lanes are populated by the real
// production writer (assimilate_validated_triplet in
// bcc32_resident_causal_constraint_participation.cuh): lane[2] always holds
// the fragment's relation/middle atom and lane[3] holds the antecedent or
// consequent endpoint atom, depending on fragment kind. Reusing
// participation::is_participation as the structural-validity predicate keeps
// this file from re-deriving that shape check a second time.
BCC32_PAGE_CONSTRAINT_HD inline bool word_participates_in_constraint_cloud(
    const rewrite::ResidentRewriteState* state, std::uint32_t word) {
  if (state == nullptr || word == 0u || word == rewrite::kInvalid)
    return false;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != participation::kFormConstraintParticipation)
      continue;
    if (!participation::is_participation(record)) continue;
    if (record.lane[2] == word || record.lane[3] == word) {
      return true;
    }
  }
  return false;
}

// Page-aware single-term access with constraint-cloud participation flagged
// alongside the raw word. `owner` is the trajectory owner exactly as used by
// trajectory_word_at / find_owned_block / ensure_trajectory_page throughout
// causal_rewrite_universe.cuh; `index` is a flat trajectory-relative event
// index (page = index / kTrajectoryPageEvents, exactly as production code
// resolves it). Unresolved indices (page absent, or beyond the last admitted
// word) return a term with `resolved == 0` and every other field zeroed.
BCC32_PAGE_CONSTRAINT_HD inline PageConstraintTerm page_constraint_term_at(
    const rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index) {
  PageConstraintTerm term{};
  if (state == nullptr) return term;
  std::uint32_t word = 0u;
  if (!rewrite::trajectory_word_at(state, owner, index, &word)) return term;
  term.word = word;
  term.resolved = 1u;
  term.participates =
      word_participates_in_constraint_cloud(state, word) ? 1u : 0u;
  return term;
}

// Bounded forward walk across every page-resolvable term of a trajectory,
// starting at `from`. Stops at the first index that trajectory_word_at
// cannot resolve (the real physical end of the admitted trajectory, however
// many pages that spans) or once `max_terms` outputs have been written,
// whichever comes first. `max_terms` bounds only the caller's output buffer;
// it is never treated as, or substituted for, a trajectory-length policy, and
// the underlying per-term resolution still has no fixed page/event ceiling
// beyond the kRecordCapacity physical scan bound already established by
// trajectory_word_at itself. Returns the number of terms written.
BCC32_PAGE_CONSTRAINT_HD inline std::uint32_t walk_page_constraint_terms(
    const rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t from, std::uint32_t max_terms, PageConstraintTerm* out) {
  if (state == nullptr || out == nullptr || max_terms == 0u) return 0u;
  std::uint32_t written = 0u;
  std::uint32_t index = from;
  while (written < max_terms) {
    const PageConstraintTerm term = page_constraint_term_at(state, owner, index);
    if (term.resolved == 0u) break;
    out[written] = term;
    ++written;
    ++index;
  }
  return written;
}

}  // namespace substrate::bcc32::resident_page_constraint_iterator
