#pragma once

// ---------------------------------------------------------------------------
// COMPARATIVE SUBSTITUTION SITE for the resident relation store.
//
// The singleton-dominance / substitution-probe audit (docs/audits/) found
// that no site in the repo retrieves a true completion's count alongside
// matched counterfactuals' counts and orders them -- every existing probe
// (count_category_mates_kernel, census_substituted_support_kernel) reads a
// single substituted lookup and thresholds or histograms it in isolation.
// This header is that missing comparative site: one small, reusable,
// PRODUCTION-CALLABLE device function over the existing table, table_counts,
// and relation_triple_lookup -- no second store, no new retrieval path, no
// invented semantics. It answers exactly one question: among a topic's one
// attested completion under a connective and a bounded set of matched
// alternatives legal for the same coordinate, does the resident count rank
// the true completion first, and by how much.
//
// This is deliberately NOT a generation/dialogue decision. It is the
// discriminability primitive a future generation-time consumer (or a
// diagnostic) can call once per candidate set; what a consumer DOES with a
// SubstitutionRankResult -- whether it becomes admissible evidence for the
// target-free output the current doctrine requires -- is that consumer's
// causal-path question, not this header's. See the Sapolskian post-land
// note on b167460: lexical/label acceptance stays diagnostic-only; this
// header exposes counts and a rank, never a word.
//
// Verified as a reusable extraction, not a new mechanism: the falsifier
// contract bcc32_cuda_resident_relation_substitution_rank_falsifier_contract
// already exercised this exact ranking arithmetic against production
// insert_relation_triple/relation_triple_lookup (compiled, linked, and run
// on-GPU: PASS, section_a_true_rank=1, section_b_true_rank=3,
// section_c_counterfactual_hits=0) before this header existed; the contract
// now calls THIS function instead of duplicating the arithmetic inline, so
// that prior GPU-verified receipt is the evidence for this extraction too.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bcc32_cuda_resident_construction_composer.cuh"

#if defined(__CUDACC__)
#define BCC32_COMPARATIVE_SUBSTITUTION_HD __host__ __device__
#else
#define BCC32_COMPARATIVE_SUBSTITUTION_HD
#endif

namespace substrate::bcc32::resident_construction {

// true_count / max_counterfactual_count / number_of_counterfactual_hits /
// true_rank_among_substitutions -- the falsifier's four statistics from the
// singleton-dominance audit, computed here rather than re-derived per call
// site.
struct SubstitutionRankResult {
  std::uint32_t true_count = 0u;
  std::uint32_t max_counterfactual_count = 0u;
  std::uint32_t number_of_counterfactual_hits = 0u;
  std::uint32_t true_rank_among_substitutions = 0u;
};

// Ranks one true substitution against a caller-supplied, bounded set of
// matched counterfactual values under the SAME (topic, connective,
// connective2) coordinate. The caller owns candidate selection (which
// alternative values are "legal for the same coordinate" is a policy
// question for the consumer, e.g. other observed fillers of the same
// connective from distinct subjects); this function owns only the
// arithmetic of comparing their resident counts through the production
// retrieval path. Competition ranking: ties share the best rank, so a
// counterfactual carrying EQUAL mass to the true completion still costs
// the true completion first place -- ge2 in the field data means "at
// least tied", not "strictly ahead".
//
// counterfactual_values must not contain true_value; a caller that cannot
// guarantee this should filter before calling, since a duplicate would
// silently double-count against itself.
[[nodiscard]] __device__ inline SubstitutionRankResult
rank_substitution_against_matched_counterfactuals(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t topic, std::uint32_t connective, std::uint32_t connective2,
    std::uint32_t true_value, const std::uint32_t* counterfactual_values,
    std::uint32_t counterfactual_count) {
  SubstitutionRankResult result;
  result.true_count = relation_triple_lookup(table, table_counts, topic,
                                             connective, connective2,
                                             true_value);
  std::uint32_t rank = 1u;
  for (std::uint32_t i = 0u; i < counterfactual_count; ++i) {
    const std::uint32_t alt = counterfactual_values[i];
    if (alt == true_value) continue;  // caller contract violation: skip, do
                                       // not let a duplicate inflate its own
                                       // rank against itself
    const std::uint32_t count = relation_triple_lookup(
        table, table_counts, topic, connective, connective2, alt);
    if (count > 0u) ++result.number_of_counterfactual_hits;
    if (count > result.max_counterfactual_count)
      result.max_counterfactual_count = count;
    if (count > result.true_count) ++rank;
  }
  result.true_rank_among_substitutions = rank;
  return result;
}

// Default candidate policy for rank_substitution_against_matched_
// counterfactuals: every DISTINCT value the store attests under the same
// (connective, connective2) from a subject other than the topic, excluding
// true_value itself. This is the ordinary "other observed fillers of the
// same relation type" reading, matching what count_category_mates_kernel
// and gather_analogical_triples_kernel already scan the table for -- same
// full-table one-thread-per-slot idiom, same kRelationTripleHashCap sweep,
// no new retrieval path.
//
// NOT gated to kRelationTripleMinCount: unlike the analogy-candidate
// gatherers, a matched counterfactual here may be a singleton -- the whole
// point of the falsifier is comparing the true completion's count against
// whatever the store actually holds for the alternatives, singleton
// included.
//
// DUPLICATE VALUES ARE POSSIBLE: if two distinct subjects both attest the
// same (connective, connective2, value), that value is appended twice. Feed
// the result through dedupe_counterfactual_values before ranking -- a
// duplicate would otherwise let one real competitor count against the true
// completion's rank more than once.
static __global__ void gather_matched_counterfactual_values_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t topic, std::uint32_t connective, std::uint32_t connective2,
    std::uint32_t true_value, std::uint32_t* candidate_values,
    std::uint32_t candidate_cap, std::uint32_t* cursor) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap || table_counts[slot] == 0u) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit) return;
  if (triple.subject == topic) return;
  if (triple.connective != connective || triple.connective2 != connective2)
    return;
  if (triple.value == true_value) return;
  const std::uint32_t position = atomicAdd(cursor, 1u);
  if (position >= candidate_cap) return;
  candidate_values[position] = triple.value;
}

// Single-thread, in-place dedup for a small bounded candidate array (call
// with <<<1,1>>> after gather_matched_counterfactual_values_kernel and
// before ranking). O(count^2), which is fine for the small bounded candidate
// sets this site is designed for -- this is caller-side list hygiene, not a
// retrieval path, so it does not need the table's own hashing discipline.
BCC32_COMPARATIVE_SUBSTITUTION_HD inline std::uint32_t
dedupe_counterfactual_values(std::uint32_t* values, std::uint32_t count) {
  std::uint32_t distinct = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    bool seen = false;
    for (std::uint32_t j = 0u; j < distinct; ++j) {
      if (values[j] == values[i]) {
        seen = true;
        break;
      }
    }
    if (!seen) values[distinct++] = values[i];
  }
  return distinct;
}

}  // namespace substrate::bcc32::resident_construction

#undef BCC32_COMPARATIVE_SUBSTITUTION_HD
