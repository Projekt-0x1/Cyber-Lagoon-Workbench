// ---------------------------------------------------------------------------
// ANSWER-SIDE CAUSAL COMPATIBILITY -- a NEW field-identity signal, own name,
// own falsifier, NO inheritance of claims from the retired one-coordinate
// question-gap learner. Defined HERE (2026-08-14, 0X1-156), not in its
// original home bcc32_resident_relation_answer_side_causal_compatibility.cuh,
// because form_witnessed_relation_plan_kernel below (in the tail this file
// includes) needs to call it, and that header itself includes THIS file --
// a real mutual dependency, not a naming accident. Defining the primitive
// where relation_triple_lookup already lives removes the cycle entirely;
// the standalone header now just documents and re-exports it so its own
// contract test and every existing include of it keep working unchanged.
//
// The prior acquisition rule for "which relation field does this question
// request" required the question's and answer's extracted 4-tuple relation
// triples to differ in EXACTLY ONE coordinate. Measured across four corpora
// spanning expository prose, children's narrative, drama, and Socratic
// dialogue -- including material chosen specifically to maximize
// question/answer parallelism -- that condition fires ZERO times in over
// 12,000 evaluated pairs; mass sits entirely at 3-4 differing coordinates.
// The mechanism was retired on its own audit-stated retirement criterion,
// not patched.
//
// This implements the audit's named replacement, unchanged in design
// intent: for each candidate field f, substitute the answer-side value into
// field f of the question-activated relation and ask whether the RESULT is
// resident-supported. The field whose substitution best closes the
// question-conditioned relation is the field the answer filled. Field
// identity comes from which substitution the store already supports, never
// from an authored reading of a wh-word -- no interrogative unit is ever
// inspected here.
// ---------------------------------------------------------------------------

struct CausalCompatibilityVote {
  std::uint32_t voted_field = kRelationFieldCount;  // sentinel: no vote
  std::uint32_t support = 0u;
  bool ambiguous = false;
};

// question_fields[0..3] = {subject, connective, connective2, value} of the
// question-activated relation, in that fixed order (matches RelationTriple's
// own layout and kRelationFieldCount == 4). One position is the unknown gap
// the answer is expected to fill; the caller need not mark which -- every
// field is tried, including ones the question already fills, so a
// substitution into an already-correct field is free to win only if it is
// genuinely the best-supported one.
[[nodiscard]] __device__ inline CausalCompatibilityVote
vote_causal_compatibility_field(const RelationTriple* table,
                                const std::uint32_t* table_counts,
                                const std::uint32_t question_fields[4],
                                std::uint32_t answer_value) {
  CausalCompatibilityVote vote;
  std::uint32_t candidate[4] = {question_fields[0], question_fields[1],
                                question_fields[2], question_fields[3]};
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    const std::uint32_t original = candidate[field];
    candidate[field] = answer_value;
    const std::uint32_t support = relation_triple_lookup(
        table, table_counts, candidate[0], candidate[1], candidate[2],
        candidate[3]);
    candidate[field] = original;  // restore before trying the next field
    if (support > vote.support) {
      vote.voted_field = field;
      vote.support = support;
      vote.ambiguous = false;
    } else if (support != 0u && support == vote.support) {
      vote.ambiguous = true;
    }
  }
  if (vote.support == 0u || vote.ambiguous)
    vote.voted_field = kRelationFieldCount;
  return vote;
}

// LIVE requested_field DERIVATION -- the actual wiring point the vote above
// was landed for and never connected, until now. `triple` is the fully-
// witnessed resident relation the reply composer already selected as its
// best match to the current question cue (an event drawn straight from
// witnessed_relation_events, never a host-authored or test-selected value).
// A field of that triple is a GAP CANDIDATE when its unit is real, open-
// class, and NOT already covered by the question's own exact cue -- the
// same filter the composer already applies when it gathers residual units
// elsewhere, so this reuses the composer's own definition of "the question
// doesn't already say this" rather than inventing a second one.
//
// Every gap-candidate position is masked with an out-of-band probe sentinel
// (never a real unit id, never kNoTripleUnit -- kNoTripleUnit is itself a
// legitimate stored value at the connective2 position for two-ary relations,
// so reusing it as an "unknown" marker there would conflate absence with
// uncertainty). Each gap candidate's OWN unit is then voted on in turn via
// vote_causal_compatibility_field above: the field it wins, at what support,
// decides which single gap the reply should lead with. No wh-word, no
// authored field name, and no answer content ever enters this decision from
// outside the resident store -- ties abstain exactly as the vote primitive's
// own falsifier requires.
[[nodiscard]] __device__ inline std::uint32_t
requested_field_from_causal_compatibility(
    const RelationTriple* table, const std::uint32_t* table_counts,
    const RelationTriple& triple, const std::uint32_t* cue_exact,
    const std::uint32_t* closed_class_mask, std::uint32_t unit_count,
    bool* ambiguous_out = nullptr) {
  if (ambiguous_out != nullptr) *ambiguous_out = false;
  if (table == nullptr || table_counts == nullptr || cue_exact == nullptr ||
      closed_class_mask == nullptr)
    return kRelationFieldCount;
  // Out of any real unit's range (unit ids are bounded by unit_count, always
  // far below this) and distinct from kNoTripleUnit itself.
  constexpr std::uint32_t kGapProbeSentinel = 0xfffffffeu;
  const std::uint32_t original[kRelationFieldCount] = {
      triple.subject, triple.connective, triple.connective2, triple.value};
  bool is_gap[kRelationFieldCount] = {};
  std::uint32_t gap_count = 0u;
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    const std::uint32_t unit = original[field];
    is_gap[field] = unit < unit_count && unit != kNoTripleUnit &&
                    closed_class_mask[unit] == 0u && cue_exact[unit] == 0u;
    if (is_gap[field]) ++gap_count;
  }
  if (gap_count == 0u) return kRelationFieldCount;
  std::uint32_t question_fields[kRelationFieldCount] = {
      original[0], original[1], original[2], original[3]};
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field)
    if (is_gap[field]) question_fields[field] = kGapProbeSentinel;

  std::uint32_t requested_field = kRelationFieldCount;
  std::uint32_t best_support = 0u;
  bool ambiguous = false;
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    if (!is_gap[field]) continue;
    const CausalCompatibilityVote vote = vote_causal_compatibility_field(
        table, table_counts, question_fields, original[field]);
    // Only a self-consistent, unambiguous vote (this candidate's own field
    // wins its own substitution test outright) counts as a real signal.
    if (vote.voted_field != field || vote.ambiguous) continue;
    if (vote.support > best_support) {
      requested_field = field;
      best_support = vote.support;
      ambiguous = false;
    } else if (vote.support != 0u && vote.support == best_support) {
      ambiguous = true;
    }
  }
  if (best_support == 0u || ambiguous) requested_field = kRelationFieldCount;
  if (ambiguous_out != nullptr) *ambiguous_out = ambiguous;
  return requested_field;
}

// LEARNED per-type directionality: for every stored triple, how much of a
// connective's triple mass is mirror-attested ((A,K,B) and (B,K,A) both
// seen)? Predication is directional; coordination is symmetric. Computed
// from the store itself -- no connective word is ever named.
static __global__ void accumulate_triple_mirror_stats_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t* type_total, std::uint32_t* type_mirrored) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap || table_counts[slot] == 0u) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit) return;
  atomicAdd(type_total + triple.connective, table_counts[slot]);
  if (relation_triple_lookup(table, table_counts, triple.value,
                             triple.connective, triple.connective2,
                             triple.subject) != 0u)
    atomicAdd(type_mirrored + triple.connective, table_counts[slot]);
}

// Occupancy diagnostic (counts only; host reads two integers for stderr).
static __global__ void count_relation_triples_kernel(
    const std::uint32_t* table_counts, std::uint32_t table_cap,
    std::uint32_t* stats) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= table_cap || table_counts[i] == 0u) return;
  atomicAdd(stats + 0u, 1u);
  atomicAdd(stats + 1u, table_counts[i]);
}

[[nodiscard]] __device__ inline bool relation_canon_eq(
    const std::uint32_t* role_canon, std::uint32_t a, std::uint32_t b) {
  if (a == b) return true;
  if (role_canon == nullptr) return false;
  return role_canon[a] == role_canon[b];
}

// A MATCHED counterfactual subject for the substitution census.
//
// The census below answers "how often does substituting the topic find
// support". On its own that number cannot distinguish a store with real
// relational structure from an episodic membership index, because a store
// where (K,X) patterns are simply common returns support for ANY subject.
// The control has to be another unit the store itself attests as a subject --
// matched on legality, chosen deterministically from the slot so the census
// is reproducible. Sampling the whole unit space instead would pick mostly
// non-subjects, make the control trivially easy to beat, and flatter the
// mechanism exactly where it needs to be doubted.
[[nodiscard]] __device__ inline std::uint32_t matched_counterfactual_subject(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t slot, std::uint32_t topic, std::uint32_t avoid,
    std::uint32_t unit_count, const std::uint32_t* role_canon) {
  std::uint32_t mixed = slot * 0x9e3779b9u;
  mixed ^= mixed >> 16u;
  mixed *= 0x85ebca6bu;
  mixed ^= mixed >> 13u;
  for (std::uint32_t step = 0u; step < kRelationTripleProbeCap; ++step) {
    const std::uint32_t candidate_slot =
        (mixed + step) & (kRelationTripleHashCap - 1u);
    if (table_counts[candidate_slot] == 0u) continue;
    const std::uint32_t candidate = table[candidate_slot].subject;
    if (candidate == kNoTripleUnit || candidate >= unit_count) continue;
    if (candidate == topic || candidate == avoid) continue;
    if (relation_canon_eq(role_canon, candidate, topic)) continue;
    return candidate;
  }
  return kNoTripleUnit;
}

// Shadow census of the SAME substituted lookup the question-time probe
// performs, over the WHOLE occupied store. The one and only difference from
// count_category_mates_kernel is that it does not build mate counts or
// mutate the store. Both paths now admit singleton source evidence; the
// production reader delays authority until two shared contexts are present.
// Keeping this diagnostic on the same seed population makes its histogram
// comparable to the actual reader rather than hiding the sparse population.
// Counts only: no mate_counts, no subject_degree, no store mutation.
static __global__ void census_substituted_support_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t topic, const std::uint32_t* role_canon,
    std::uint32_t unit_count, std::uint32_t* census_histogram = nullptr,
    std::uint32_t* census_total = nullptr,
    std::uint32_t* census_source_singletons = nullptr,
    // Matched-counterfactual arm. Defaulted off, so every existing launch is
    // byte-for-byte unchanged; pass them to get the discrimination statistic.
    std::uint32_t* counterfactual_histogram = nullptr,
    std::uint32_t* counterfactual_total = nullptr,
    std::uint32_t* topic_strictly_greater = nullptr,
    std::uint32_t* counterfactual_strictly_greater = nullptr,
    std::uint32_t* support_ties = nullptr) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap) return;
  // Occupancy, not evidence: an empty slot holds no relation to substitute
  // into. This is the only count test the census applies.
  if (table_counts[slot] == 0u) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit || triple.subject >= unit_count) return;
  if (triple.subject == topic) return;
  if (relation_canon_eq(role_canon, triple.subject, topic)) return;
  const std::uint32_t probe_support = relation_triple_lookup(
      table, table_counts, topic, triple.connective, triple.connective2,
      triple.value);
  if (census_total != nullptr) atomicAdd(census_total, 1u);
  if (census_histogram != nullptr)
    atomicAdd(census_histogram + (probe_support < kRelationProbeSupportBins - 1u
                                      ? probe_support
                                      : kRelationProbeSupportBins - 1u),
              1u);
  // How much of this census the production gate would have discarded: the
  // once-attested sources the probe never reaches.
  if (census_source_singletons != nullptr &&
      table_counts[slot] < kRelationTripleMinCount)
    atomicAdd(census_source_singletons, 1u);

  // The falsifier, evaluated on the SAME slot and the SAME (K, K2, X) so the
  // only thing that varies is which subject is substituted. If the topic and
  // a matched alternative subject find support at the same rate, the store is
  // an episodic membership index -- it records that (K,X) occurred, not
  // anything about the topic -- and no amount of threshold tuning makes it a
  // relation model. Reported as raw tallies, never as a verdict: a verdict
  // computed device-side would be a host-authored conclusion about resident
  // matter.
  if (counterfactual_histogram == nullptr && counterfactual_total == nullptr &&
      topic_strictly_greater == nullptr &&
      counterfactual_strictly_greater == nullptr && support_ties == nullptr)
    return;
  const std::uint32_t alternative = matched_counterfactual_subject(
      table, table_counts, slot, topic, triple.subject, unit_count, role_canon);
  if (alternative == kNoTripleUnit) return;
  const std::uint32_t counterfactual_support = relation_triple_lookup(
      table, table_counts, alternative, triple.connective, triple.connective2,
      triple.value);
  if (counterfactual_total != nullptr) atomicAdd(counterfactual_total, 1u);
  if (counterfactual_histogram != nullptr)
    atomicAdd(counterfactual_histogram +
                  (counterfactual_support < kRelationProbeSupportBins - 1u
                       ? counterfactual_support
                       : kRelationProbeSupportBins - 1u),
              1u);
  if (probe_support > counterfactual_support) {
    if (topic_strictly_greater != nullptr) atomicAdd(topic_strictly_greater, 1u);
  } else if (counterfactual_support > probe_support) {
    if (counterfactual_strictly_greater != nullptr)
      atomicAdd(counterfactual_strictly_greater, 1u);
  } else if (support_ties != nullptr) {
    atomicAdd(support_ties, 1u);
  }
}

// ---------------------------------------------------------------------------
// DISTRIBUTIONAL CATEGORY + PROPORTIONAL ANALOGY over the resident relation
// store. The adjacency wall (proven on the real gate): a generation walk over
// ATTESTED triples is corpus adjacency, so locally coherent output
// reconstructs verbatim corpus spans, while suppressing that copies forces
// topic drift. The escape is the classical poverty-of-stimulus move: learn
// CATEGORIES from shared predication contexts and generalize. Two units
// belong to the same emergent category exactly when the store itself proves
// they take the same predicates with the same arguments -- no authored word
// classes, no features, only counts already in conserved resident matter:
//
//   mate(T, A') = |{(K,X) : count(T,K,X) >= seed AND
//                            count(A',K,X) >= seed}|
//
// A mate is a category candidate only after the reader observes at least two
// such shared contexts. A claim (T, K, B') is LICENSED-NOVEL when some
// category mate A' of T
// attests (A', K, B') while (T, K, B') itself is UNATTESTED. Such a clause is
// (a) surface-novel by construction -- the byte string never occurs in the
// corpus, so it cannot extend a verbatim span -- and (b) role-licensed, so it
// stays a well-formed predication about T. This is "its own words" in the
// literal sense: the organism asserts what its category structure entails,
// not what the corpus adjacency dictates. Both kernels run at retrieval time
// over the existing table; nothing new is stored and no mass moves.
// ---------------------------------------------------------------------------
