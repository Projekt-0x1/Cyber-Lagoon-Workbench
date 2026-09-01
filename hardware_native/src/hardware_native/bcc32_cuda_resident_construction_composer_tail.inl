inline constexpr std::uint32_t kAnalogyFlag = 0x40000000u;
inline constexpr std::uint32_t kAnalogyMinSharedContexts = 2u;

static __global__ void count_category_mates_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t topic, const std::uint32_t* role_canon,
    std::uint32_t unit_count, std::uint32_t* mate_counts,
    std::uint32_t* subject_degree,
    std::uint32_t* probe_support_histogram = nullptr,
    std::uint32_t* probe_total = nullptr) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap) return;
  // Keep singleton observations in the structural census. They are seeds,
  // not standalone answers: category membership below still requires two
  // independently shared contexts.
  if (table_counts[slot] < kRelationTripleSeedCount) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit || triple.subject >= unit_count) return;
  // Context degree: how many attested claims this unit heads at all. The
  // normalizer that separates a genuine category mate (large shared
  // FRACTION) from a hub that overlaps everything a little.
  if (subject_degree != nullptr)
    atomicAdd(subject_degree + triple.subject, 1u);
  if (triple.subject == topic) return;
  if (relation_canon_eq(role_canon, triple.subject, topic)) return;
  // Shared predication context: the topic attests the SAME (K, X) claim.
  // The support this substituted triple actually carries in the store. The
  // probed subset is not the store at large: the SOURCE slot is already
  // gated to count >= seed, so the global count distribution does not
  // predict what these probes return. Qualification still happens at the
  // two-shared-context gate in the analogical reader.
  const std::uint32_t probe_support = relation_triple_lookup(
      table, table_counts, topic, triple.connective, triple.connective2,
      triple.value);
  if (probe_total != nullptr) atomicAdd(probe_total, 1u);
  if (probe_support_histogram != nullptr)
    atomicAdd(probe_support_histogram +
                  (probe_support < kRelationProbeSupportBins - 1u
                       ? probe_support
                       : kRelationProbeSupportBins - 1u),
              1u);
  if (probe_support >= kRelationTripleSeedCount)
    atomicAdd(mate_counts + triple.subject, 1u);
}

// Structural category formation for sparse natural contact.  The old reader
// compared complete (subject, connective, value) triples, which makes two
// subjects look unrelated whenever their observed objects differ.  This pass
// compares only the recurrent connective pattern already formed by the adult.
// It does not erase object-specific facts: those remain in RelationTriple and
// are used by the analogical reader to compose a novel value for the query
// subject.  A mate still needs two independently shared role contexts.
static __global__ void count_structural_category_mates_kernel(
    const RelationRole* role_table, const std::uint32_t* role_counts,
    std::uint32_t topic, std::uint32_t unit_count,
    std::uint32_t* mate_counts, std::uint32_t* subject_degree,
    std::uint32_t* support_histogram = nullptr,
    std::uint32_t* support_total = nullptr) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationRoleHashCap || role_counts[slot] < kRelationRoleSeedCount)
    return;
  const RelationRole role = role_table[slot];
  if (role.subject == kNoTripleUnit || role.subject >= unit_count ||
      role.subject == topic)
    return;
  if (subject_degree != nullptr) atomicAdd(subject_degree + role.subject, 1u);
  const std::uint32_t support = relation_role_lookup(
      role_table, role_counts, topic, role.connective, role.connective2);
  if (support_total != nullptr) atomicAdd(support_total, 1u);
  if (support_histogram != nullptr)
    atomicAdd(support_histogram +
                  (support < kRelationProbeSupportBins - 1u
                       ? support
                       : kRelationProbeSupportBins - 1u),
              1u);
  if (support >= kRelationRoleSeedCount)
    atomicAdd(mate_counts + role.subject, 1u);
}

static __global__ void gather_analogical_triples_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t topic, const std::uint32_t* mate_counts,
    const std::uint32_t* role_canon, std::uint32_t unit_count,
    std::uint32_t* candidates, std::uint32_t* cursor) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap) return;
  if (table_counts[slot] < kRelationTripleSeedCount) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit || triple.subject >= unit_count) return;
  if (triple.subject == topic) return;
  if (mate_counts[triple.subject] < kAnalogyMinSharedContexts) return;
  if (triple.value == topic ||
      relation_canon_eq(role_canon, triple.value, topic))
    return;
  // Surface novelty: the composed claim (topic, K, B') must be UNATTESTED --
  // otherwise it is ordinary retrieval and belongs to the attested channel.
  if (relation_triple_lookup(table, table_counts, topic, triple.connective,
                             triple.connective2, triple.value) != 0u)
    return;
  const std::uint32_t position = atomicAdd(cursor, 1u);
  if (position >= kRelationTripleCandidateCap) return;
  candidates[position] = slot | kAnalogyFlag;
}

[[nodiscard]] __device__ inline bool relation_cue_exact(
    const std::uint32_t* role_canon, const std::uint32_t* cue_exact,
    std::uint32_t unit) {
  if (cue_exact == nullptr) return false;
  if (cue_exact[unit] != 0u) return true;
  return role_canon != nullptr && cue_exact[role_canon[unit]] != 0u;
}

// A unit is a learned interrogative opener when it repeatedly starts a
// resident segment whose final unit contains '?'. This discovers the
// question-onset category from ordinary stream experience; no wh-word list
// or authored language identity participates.
static __global__ void learn_qonset_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* role_canon, std::uint32_t* qonset_count,
    const std::uint64_t* contact_evidence_revision,
    std::uint64_t* qonset_evidence_revision,
    std::uint32_t unit_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sequence_count) return;
  const std::uint32_t terminal = sequence[i];
  if (terminal >= unit_count) return;
  bool interrogative = false;
  bool declarative_terminal = false;
  for (std::uint32_t b = 0u; b < unit_lengths[terminal]; ++b) {
    const std::uint32_t byte =
        construction_unit_byte(unit_content, unit_words, terminal, b);
    if (byte == kQonsetQuestionByte) {
      interrogative = true;
    }
    declarative_terminal |=
        byte == static_cast<std::uint32_t>('.') ||
        byte == static_cast<std::uint32_t>('!') ||
        byte == static_cast<std::uint32_t>('\n');
  }
  if (!interrogative && !declarative_terminal) return;
  std::uint32_t onset = i;
  while (onset != 0u && segment_ids[onset - 1u] == segment_ids[i]) {
    const std::uint32_t previous = sequence[onset - 1u];
    bool closes_previous = false;
    if (previous < unit_count) {
      for (std::uint32_t b = 0u; b < unit_lengths[previous]; ++b) {
        const std::uint32_t byte =
            construction_unit_byte(unit_content, unit_words, previous, b);
        if (byte == static_cast<std::uint32_t>('.') ||
            byte == static_cast<std::uint32_t>('!') ||
            byte == kQonsetQuestionByte) {
          closes_previous = true;
          break;
        }
      }
    }
    if (closes_previous) break;
    --onset;
  }
  std::uint32_t opener = sequence[onset];
  if (opener >= unit_count) return;
  (void)role_canon;
  if (interrogative) {
    atomicAdd(qonset_count + opener, 1u);
    if (contact_evidence_revision != nullptr &&
        qonset_evidence_revision != nullptr)
      atomicMax(
          reinterpret_cast<unsigned long long*>(
              qonset_evidence_revision + opener),
          static_cast<unsigned long long>(contact_evidence_revision[0]));
  } else {
    (void)decrement_question_onset(qonset_count + opener);
  }
}

// Locate the learned relation-operator position in the current cue. For a
// direct question this is the unit immediately after a learned interrogative
// opener. The result remains absent for unfamiliar forms, preserving the
// evidence-ranked fallback rather than inventing an operator.
static __global__ void derive_relation_operator_order_kernel(
    const std::uint32_t* cue_scores, const std::uint32_t* cue_orders,
    const std::uint32_t* role_canon, const std::uint32_t* qonset_count,
    const std::uint32_t* relation_type_total,
    std::uint32_t cue_identity_floor, std::uint32_t unit_count,
    std::uint32_t* operator_order) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  std::uint32_t canon = unit;
  if (role_canon != nullptr && role_canon[unit] < unit_count)
    canon = role_canon[unit];
  std::uint32_t cue_weight = cue_scores[unit];
  if (cue_scores[canon] > cue_weight) cue_weight = cue_scores[canon];
  if (cue_weight < cue_identity_floor) return;
  std::uint32_t order = cue_orders[unit];
  if (cue_orders[canon] < order) order = cue_orders[canon];
  if (order == kNoTripleUnit) return;
  if (qonset_count != nullptr &&
      qonset_count[unit] >= kQonsetTopicFloor) {
    if (order != kNoTripleUnit - 1u) atomicMin(operator_order, order + 1u);
    return;
  }
  std::uint32_t type_mass = relation_type_total[unit];
  if (relation_type_total[canon] > type_mass)
    type_mass = relation_type_total[canon];
  if (type_mass == 0u) return;
  atomicMin(operator_order, order);
}

static __global__ void derive_latest_exact_content_order_kernel(
    const std::uint32_t* cue_exact, const std::uint32_t* cue_orders,
    const std::uint32_t* closed_class_mask, std::uint32_t unit_count,
    std::uint32_t* latest_order) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count || cue_exact[unit] == 0u ||
      closed_class_mask[unit] != 0u || cue_orders[unit] == kNoTripleUnit)
    return;
  atomicMax(latest_order, cue_orders[unit]);
}

// COMMIT step 1: latch the topic -- the strongest legal subject-field unit,
// the identical rule every commitment variant uses, so channel ON/OFF differ
// only in what predicates. meta[2] carries the topic to the gather/form
// kernels; meta[0]/[1]/[3] reset.
static __global__ void select_triple_topic_kernel(
    const std::uint32_t* subject_ids, const std::uint32_t* subject_weights,
    const std::uint32_t* subject_count, std::uint32_t subject_cap,
    const std::uint32_t* closed_class_mask, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* boundary_mask,
    const std::uint32_t* filler_terminal_mask,
    const std::uint32_t* role_canon, std::uint32_t unit_count,
    std::uint32_t* meta) {
  extern __shared__ std::uint32_t reduction[];
  std::uint32_t* best_weights = reduction;
  std::uint32_t* best_indices = best_weights + blockDim.x;
  std::uint32_t* best_topics = best_indices + blockDim.x;
  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  const std::uint32_t lane = threadIdx.x;
  std::uint32_t local_weight = 0u;
  std::uint32_t local_index = subjects;
  std::uint32_t local_topic = kNoTripleUnit;
  for (std::uint32_t i = lane; i < subjects; i += blockDim.x) {
    std::uint32_t unit = subject_ids[i];
    if (!commitment_filler_legal(unit, closed_class_mask, unit_lengths,
                                 unit_content, unit_words, boundary_mask,
                                 filler_terminal_mask)) {
      // A cue word that reached the field carrying trailing punctuation
      // ("revolution.", "work?") fails filler hygiene as-is, but its
      // learned canon representative -- the case-folded CORE byte form the
      // canon organ already computed -- is the same word cleanly bounded.
      // Latch through the canon instead of dropping the strongest topic
      // and letting a weaker function-word residue win the latch.
      if (role_canon == nullptr || unit >= unit_count) continue;
      const std::uint32_t representative = role_canon[unit];
      if (representative == unit || representative >= unit_count) continue;
      if (!commitment_filler_legal(representative, closed_class_mask,
                                   unit_lengths, unit_content, unit_words,
                                   boundary_mask, filler_terminal_mask))
        continue;
      unit = representative;
    }
    if (subject_weights[i] > local_weight ||
        (subject_weights[i] == local_weight && i < local_index)) {
      local_weight = subject_weights[i];
      local_index = i;
      local_topic = unit;
    }
  }
  best_weights[lane] = local_weight;
  best_indices[lane] = local_index;
  best_topics[lane] = local_topic;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride &&
        (best_weights[lane] < best_weights[lane + stride] ||
         (best_weights[lane] == best_weights[lane + stride] &&
          best_indices[lane] > best_indices[lane + stride]))) {
      best_weights[lane] = best_weights[lane + stride];
      best_indices[lane] = best_indices[lane + stride];
      best_topics[lane] = best_topics[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0u) {
    meta[0] = 0u;
    meta[1] = 0u;
    meta[2] = best_indices[0] < subjects ? best_topics[0] : kNoTripleUnit;
    meta[3] = 0u;
  }
}

// COMMIT step 2: parallel retrieval over the resident table. A slot is a
// candidate when its subject (forward claim about the topic) or its value
// (reverse claim whose object is the topic) canon-matches the topic. A slot
// matching on both ends is a self-claim and is dropped.
static __global__ void gather_relation_triples_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    const std::uint32_t* role_canon, const std::uint32_t* cue_scores,
    const std::uint32_t* cue_orders, const std::uint32_t* cue_exact,
    const std::uint32_t* operator_order,
    const std::uint32_t* required_topic_unit,
    std::uint32_t cue_identity_floor,
    std::uint32_t* candidates, std::uint32_t* cursor) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationTripleHashCap || table_counts[slot] == 0u) return;
  const RelationTriple triple = table[slot];
  if (triple.subject == kNoTripleUnit) return;
  // A relation whose complete A-K-B surface is already present in the cue is
  // a restatement of that cue, not evidence recovered from prior matter.  It
  // can arise when a repeated question was observed earlier; admitting it
  // makes the query replay itself as its own answer.  This is identity-only
  // exclusion over the resident cue field, not a host question rule.
  if (cue_exact != nullptr && cue_exact[triple.subject] != 0u &&
      cue_exact[triple.connective] != 0u && cue_exact[triple.value] != 0u)
    return;
  // A learned question form may name an operator which has never appeared in
  // a factual episode (for example, a newly acquired interrogative frame over
  // an older action relation).  It can rank an otherwise grounded claim, but
  // cannot be a lexical admission gate: requiring the same surface unit here
  // reduces a novel question to replay of previously co-located Q->A text.
  // The later commitment tournament retains the exact cue/operator evidence
  // as a preference while all exact endpoint matches remain available.
  std::uint32_t subject_cue = cue_scores[triple.subject];
  std::uint32_t value_cue = cue_scores[triple.value];
  if (role_canon != nullptr) {
    const std::uint32_t subject_rep = role_canon[triple.subject];
    const std::uint32_t value_rep = role_canon[triple.value];
    if (cue_scores[subject_rep] > subject_cue) subject_cue = cue_scores[subject_rep];
    if (cue_scores[value_rep] > value_cue) value_cue = cue_scores[value_rep];
  }
  bool subject_matches = subject_cue >= cue_identity_floor;
  bool value_matches = value_cue >= cue_identity_floor;
  if (required_topic_unit != nullptr) {
    // A required endpoint is an exact cue-to-fact bridge.  Canonical
    // neighbors remain usable in the later ranking geometry, but admitting
    // them here lets a generic role class impersonate a freshly learned
    // entity and opens unrelated corpus triples.
    subject_matches = triple.subject == required_topic_unit[0];
    value_matches = triple.value == required_topic_unit[0];
  }
  if (subject_matches == value_matches) return;  // no match, or cue-only restatement
  const bool forward = subject_matches;
  const std::uint32_t position = atomicAdd(cursor, 1u);
  if (position >= kRelationTripleCandidateCap) return;
  candidates[position] = slot | (forward ? 0u : 0x80000000u);
}

// COMMIT step 3: rank the gathered triples and write the PROPOSITIONAL plan
// -- ordered (subject, connective[, connective2], value) clauses, at most
// kRelationTripleMaxClauses. Rank = attested count, vitality-normalized on
// the claim's non-topic end (the same resident PMI analogue every
// commitment variant uses) so generic filler values lose to specific ones.
// Evidence tiers: count >= kRelationTripleMinCount first; when no clause
// forms at that bar, single-attestation triples are admitted (still learned
// evidence; the tier is surfaced in meta[3] for honest diagnostics). The
// non-topic end must pass the same filler hygiene + near-duplicate guards
// as every commitment; both connectives must still be glue under the
// current partition, so the rendered clause reads subject-predicate-value.
template <class Bigram>
static __global__ void form_triple_commitment_kernel(
    const RelationTriple* table, const std::uint32_t* table_counts,
    const std::uint32_t* candidates, const std::uint32_t* cursor,
    const std::uint32_t* type_total, const std::uint32_t* type_mirrored,
    const std::uint32_t* cue_scores, const std::uint32_t* cue_orders,
    const std::uint32_t* cue_exact, std::uint32_t* operator_order,
    std::uint32_t cue_identity_floor,
    bool topic_fallback,
    const std::uint32_t* role_canon, const std::uint32_t* qonset_count,
    std::uint32_t unit_count, const std::uint32_t* unit_vitality,
    const std::uint32_t* closed_class_mask, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* boundary_mask,
    const std::uint32_t* filler_terminal_mask,
    const Bigram* phrase_bigrams, const std::uint32_t* phrase_bigram_counts,
    std::uint32_t phrase_bigram_count,
    // Reply-scope refractory state (thematic progression): endpoints already
    // uttered earlier in THIS reply. A candidate whose new-information end
    // near-duplicates one of them is suppressed, so a chained reply moves to
    // new matter instead of restating. nullptr = single-shot commitment,
    // exact prior behavior. When non-null the kernel is realizing a
    // SELF-CONTINUATION: the topic is the organism's own just-uttered rheme
    // (exact-unit gated upstream), so content-word predicates attested in
    // the store are admissible under the ordinary count tier -- the
    // "cue-typed only" restriction exists to stop promiscuous untyped
    // retrieval from an external cue, and the exact self-topic plus the
    // refractory set provide that discipline here.
    const std::uint32_t* prior_ends, std::uint32_t prior_end_count,
    // Relation-type carry (thematic progression preserves the rhetorical
    // relation, not just the topic). When != kNoTripleUnit, a self-chain
    // candidate's connective must canon-match this type: a "cause" answer
    // continues as a causal chain, a copular answer as further predication,
    // instead of drifting into the theme's highest-count collocate. This is
    // the argument-structure constraint the external cue supplies for the
    // first sentence, propagated by the organism's own prior commitment.
    std::uint32_t required_connective_canon,
    // Persistent topic field (prefrontal-style sustained goal representation).
    // For a self-continuation, the new-information end must retain associative
    // mass with the ORIGINAL question topic above topic_affinity_floor, so the
    // paragraph keeps developing ONE subject instead of random-walking into
    // the local collocate of whatever was said last. nullptr disables the
    // tether (first-sentence answer, governed by the external cue).
    const unsigned long long* topic_affinity,
    unsigned long long topic_affinity_floor,
    // When a candidate carries kAnalogyFlag its clause is the LICENSED-NOVEL
    // composition (analogical_topic, K, value): the stored triple supplies
    // predicate and value, the topic substitutes as subject (proportional
    // analogy over the emergent category, see the analogy kernels above).
    std::uint32_t analogical_topic,
    // Category-evidence strength per unit (shared-context counts from
    // count_category_mates_kernel). Folded multiplicatively into an
    // analogical candidate's rank so a claim licensed by a STRONG category
    // mate (many shared predication contexts) outranks one licensed by a
    // marginal mate -- sparsity is handled by ranking, not a hard cliff.
    const std::uint32_t* mate_counts,
    const std::uint32_t* subject_degree,
    // Emergent word-class field (0=OTHER 1=FUNC 2=NOUN 3=VERB), nullable.
    // In self-chain mode a FUNC-classed value cannot be the new information
    // of a clause ("would all"), and a value carrying control bytes is a
    // fused cross-line sensor artifact, not a word.
    const std::uint8_t* unit_pos,
    std::uint32_t max_clauses, std::uint32_t* plan,
    std::uint32_t* meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  meta[0] = 0u;             // plan extent (reset here, not only in the topic
  meta[1] = 0u;             // kernel: analogy retrieval skips that kernel,
  meta[3] = 0u;             // and a stale extent would replay the old plan)
  meta[2] = kNoTripleUnit;
  meta[4] = kNoTripleUnit;  // rheme (new-information end) of the last clause
  meta[5] = 0u;             // orientation of the last clause (1 = forward)
  meta[6] = kNoTripleUnit;  // connective (canon) of the last clause
  meta[7] = kNoTripleUnit;  // retained table slot of the sole source clause
  // CUE-TYPED RELATION SELECTION: the question itself carries its relation
  // type -- the glue units of the CURRENT cue ("what IS talent" carries
  // 'is'), read from the persisted cue near-identity field. A candidate
  // whose connective the cue actually said answers in the asked relation
  // and takes strict priority. The type is supplied by the cue at question
  // time, never authored here.
  const std::uint32_t candidate_count =
      cursor[0] < kRelationTripleCandidateCap ? cursor[0]
                                              : kRelationTripleCandidateCap;
  if (candidate_count == 0u) return;
  std::uint32_t asked_operator_order =
      !topic_fallback && operator_order != nullptr ? operator_order[0]
                                                    : kNoTripleUnit;
  // Some streams contain too few repeated question episodes to establish a
  // durable opener category. In that case, infer the earliest cue-ordered
  // unit that the resident store itself already uses as a relation type.
  // This is a structural fallback over learned triples, not a word list.
  if (!topic_fallback && asked_operator_order == kNoTripleUnit &&
      cue_orders != nullptr &&
      cue_scores != nullptr) {
    for (std::uint32_t c = 0u; c < candidate_count; ++c) {
      const std::uint32_t slot = candidates[c] & ~0x80000000u;
      const std::uint32_t connective = table[slot].connective;
      std::uint32_t canon = connective;
      if (role_canon != nullptr && role_canon[connective] < unit_count)
        canon = role_canon[connective];
      if (qonset_count != nullptr && qonset_count[canon] >= kQonsetTopicFloor)
        continue;
      std::uint32_t cue_weight = cue_scores[connective];
      if (cue_scores[canon] > cue_weight) cue_weight = cue_scores[canon];
      if (cue_weight < cue_identity_floor) continue;
      std::uint32_t order = cue_orders[connective];
      if (cue_orders[canon] < order) order = cue_orders[canon];
      if (order < asked_operator_order) asked_operator_order = order;
    }
  }
  if (operator_order != nullptr) operator_order[0] = asked_operator_order;
  std::uint32_t committed_ends[2u * kRelationTripleMaxClauses];
  std::uint32_t committed_end_count = 0u;
  std::uint32_t used_connectives[kRelationTripleMaxClauses];
  std::uint32_t extent = 0u;
  std::uint32_t clauses = 0u;
  std::uint32_t tier = kRelationTripleMinCount;
  const bool self_chain = prior_ends != nullptr;
  const std::uint32_t clause_cap = max_clauses < kRelationTripleMaxClauses
                                       ? max_clauses
                                       : kRelationTripleMaxClauses;
  while (clauses < clause_cap && extent + 4u <= kCommitmentCap) {
    std::uint32_t best = kNoTripleUnit;
    unsigned long long best_strength = 0ull;
    unsigned long long best_count = 0ull;
    std::uint32_t best_topic = kNoTripleUnit;
    std::uint32_t best_end = kNoTripleUnit;
    std::uint32_t best_topic_weight = 0u;
    std::uint32_t best_topic_order = 0u;
    std::uint32_t best_cue_weight = 0u;
    bool best_forward = false;
    bool best_operator_typed = false;
    bool best_direct_topic = false;
    bool best_exact_topic = false;
    bool best_exact_endpoint = false;
    for (std::uint32_t c = 0u; c < candidate_count; ++c) {
      const std::uint32_t entry = candidates[c];
      const bool analogical = (entry & kAnalogyFlag) != 0u;
      const std::uint32_t slot = entry & ~(0x80000000u | kAnalogyFlag);
      const bool forward = (entry & 0x80000000u) == 0u;
      const std::uint32_t count = table_counts[slot];
      const RelationTriple triple = table[slot];
      const std::uint32_t candidate_topic =
          analogical ? analogical_topic
                     : (forward ? triple.subject : triple.value);
      const std::uint32_t other = forward ? triple.value : triple.subject;
      const std::uint32_t uttered_subject =
          analogical ? analogical_topic : triple.subject;
      std::uint32_t topic_canon = candidate_topic;
      if (role_canon != nullptr && candidate_topic < unit_count &&
          role_canon[candidate_topic] < unit_count)
        topic_canon = role_canon[candidate_topic];
      if (qonset_count != nullptr && topic_canon < unit_count &&
          qonset_count[topic_canon] >= kQonsetTopicFloor)
        continue;
      if (relation_canon_eq(role_canon, other, candidate_topic)) continue;
      // LEARNED directionality filter: a connective TYPE whose mirror-
      // attested mass crosses the threshold behaves as symmetric
      // coordination and cannot type a claim; and a pair whose own mirror
      // is attested is coordination regardless of its type.
      if ((static_cast<unsigned long long>(type_mirrored[triple.connective])
           << kRelationTripleMirrorShift) >=
          static_cast<unsigned long long>(type_total[triple.connective]))
        continue;
      if (relation_triple_lookup(table, table_counts, triple.value,
                                 triple.connective, triple.connective2,
                                 uttered_subject) != 0u)
        continue;
      if (!commitment_filler_legal(other, closed_class_mask, unit_lengths,
                                   unit_content, unit_words, boundary_mask,
                                   filler_terminal_mask))
        continue;
      if (commitment_near_duplicate(other, committed_ends, committed_end_count,
                                    unit_lengths, unit_content, unit_words))
        continue;
      if (self_chain &&
          commitment_near_duplicate(other, prior_ends, prior_end_count,
                                    unit_lengths, unit_content, unit_words))
        continue;  // reply-scope refractory: already said in this reply
      {
        // A value that near-duplicates its own connective renders as a
        // stutter ("all all", "over over") -- a degenerate learned triple
        // from a noisy content/glue partition, never a claim worth uttering.
        std::uint32_t connective_ends[2] = {
            triple.connective,
            triple.connective2 == kNoTripleUnit ? triple.connective
                                                : triple.connective2};
        if (commitment_near_duplicate(other, connective_ends, 2u,
                                      unit_lengths, unit_content, unit_words))
          continue;
      }
      {
        // Surface-artifact ban, ALL channels (attested included): a value
        // carrying control bytes, quotes, brackets, or interior periods
        // renders as fused garbage or unbalanced closure; and a value with
        // NO alphabetic byte at all ("1945") is not a word -- a clause
        // whose new information is bare digits carries no claim.
        bool artifact = false;
        bool has_alpha = false;
        for (std::uint32_t o = 0u;
             o < unit_lengths[other] && !artifact; ++o) {
          const std::uint32_t b =
              construction_unit_byte(unit_content, unit_words, other, o);
          artifact = b < 0x20u || b == 0x22u || b == 0x28u ||
                     b == 0x29u || b == 0x5bu || b == 0x5du ||
                     b == 0x7bu || b == 0x7du || b == 0xe2u;
          has_alpha |= (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z');
        }
        if (artifact || !has_alpha) continue;
      }
      {
        // The SUBJECT opens the clause and the CONNECTIVES glue it; any of
        // them carrying clause-breaking bytes renders degenerate surface:
        // a subject "talent?" becomes a 1-word clause (and, re-selectable
        // after rotation, the SAME degenerate clause again -- duplicate),
        // a connective "(the" leaves an unbalanced paren. Ban control
        // bytes, closure punctuation, quotes and brackets fused into these
        // plan roles (byte-class hygiene; the apostrophe stays legal for
        // possessives).
        const std::uint32_t plan_roles[3] = {
            uttered_subject, triple.connective,
            triple.connective2 == kNoTripleUnit ? triple.connective
                                                : triple.connective2};
        bool artifact = false;
        for (std::uint32_t r = 0u; r < 3u && !artifact; ++r) {
          const std::uint32_t role_unit = plan_roles[r];
          if (role_unit >= unit_count) continue;
          for (std::uint32_t o = 0u;
               o < unit_lengths[role_unit] && !artifact; ++o) {
            const std::uint32_t b = construction_unit_byte(
                unit_content, unit_words, role_unit, o);
            artifact = b < 0x20u || b == 0x21u || b == 0x22u || b == 0x28u ||
                       b == 0x29u || b == 0x2eu || b == 0x3au || b == 0x3bu ||
                       b == 0x3fu || b == 0x5bu || b == 0x5du || b == 0x7bu ||
                       b == 0x7du || b == 0xe2u;
          }
        }
        if (artifact) continue;
      }
      if (self_chain) {
        // QUALITY GATE for self-continuation (no external cue types the
        // relation, so attestation and substance carry the discipline the
        // cue provides in the first sentence). Require: (a) the new-
        // information end is a substantial content word, not a stray
        // short token a noisy content/glue partition let through
        // ("how", "why", "all", "each"); (b) the whole triple is attested
        // more than once -- a repeated directed bridge is a claim, a
        // one-off collocation is not. These reject the confabulation that
        // an unshackled associator produces while keeping genuine learned
        // predications ("changes cause growth", "revolution was social").
        if (unit_lengths[other] < kConstructionMinFillerBytes) continue;
        if (unit_lengths[uttered_subject] < kConstructionMinFillerBytes)
          continue;
        if (analogical) {
          // A singleton completion may be used only after its subject has
          // earned two shared resident contexts. This prevents one-off
          // co-occurrence from becoming an answer while allowing sparse
          // facts to participate in a genuinely distributed abstraction.
          if (mate_counts == nullptr ||
              mate_counts[triple.subject] < kAnalogyMinSharedContexts)
            continue;
        } else if (count < kRelationTripleMinCount) {
          continue;
        }
        if (unit_pos != nullptr && unit_pos[other] == 1u) continue;  // FUNC
        // ORDER-PRESERVING FLUENCY: a continuation clause must be bound by
        // learned GLUE ("is", "of", "was", "and") -- the claim stays an
        // ordered subject-glue-value island that reads as grammar. Content-K
        // bridges ("growth supply beyond", "talent Yet guarantee") are noun
        // collisions, the measured source of the telegraphic broken surface,
        // and are only trustworthy when an external cue types them.
        const bool self_glue =
            closed_class_mask[triple.connective] != 0u &&
            (triple.connective2 == kNoTripleUnit ||
             closed_class_mask[triple.connective2] != 0u);
        if (!self_glue) continue;
        {
          // Bar control bytes (fused cross-line artifacts) AND quote/paren
          // carriers: an unpaired quote or bracket in a composed clause
          // fails balanced closure, and the pair can never be completed by
          // a unit-recombination generator.
          bool artifact = false;
          for (std::uint32_t o = 0u;
               o < unit_lengths[other] && !artifact; ++o) {
            const std::uint32_t b =
                construction_unit_byte(unit_content, unit_words, other, o);
            artifact = b < 0x20u || b == 0x21u || b == 0x22u || b == 0x27u ||
                       b == 0x28u || b == 0x29u || b == 0x2eu || b == 0x3au ||
                       b == 0x3bu || b == 0x3fu || b == 0x5bu || b == 0x5du ||
                       b == 0x7bu || b == 0x7du || b == 0xe2u;
          }
          if (artifact) continue;
        }
        // TOPICAL TETHER: the new information must still belong to the
        // original topic's associative field. This is the long-range
        // constraint that turns a local associator into a topic-developing
        // one -- the "forest" the bare collocation walk misses.
        if (topic_affinity != nullptr) {
          unsigned long long affinity = topic_affinity[other];
          if (role_canon != nullptr && role_canon[other] < unit_count &&
              topic_affinity[role_canon[other]] > affinity)
            affinity = topic_affinity[role_canon[other]];
          if (affinity < topic_affinity_floor) continue;
        }
        // RELATION-TYPE CARRY: the continuation must express the SAME
        // rhetorical relation as the sentence it continues (canon-matched
        // connective), so the paragraph develops one line of argument
        // instead of hopping to the theme's top collocate. This is the
        // deep-structure constraint (which relation binds the arguments),
        // not a surface co-occurrence.
        if (required_connective_canon != kNoTripleUnit) {
          std::uint32_t connective_canon = triple.connective;
          if (role_canon != nullptr && triple.connective < unit_count &&
              role_canon[triple.connective] < unit_count)
            connective_canon = role_canon[triple.connective];
          if (connective_canon != required_connective_canon) continue;
        }
      }
      bool connective_used = false;
      for (std::uint32_t k = 0u; k < clauses; ++k)
        connective_used |= used_connectives[k] == triple.connective;
      if (clauses != 0u && connective_used) continue;  // vary the predicate
      // The connective's cue near-identity score (canonical-pooled): only
      // words the CURRENT question actually said clear the identity floor,
      // so stale field residue cannot type the relation.
      std::uint32_t cue_weight = 0u;
      if (cue_scores != nullptr) {
        cue_weight = cue_scores[triple.connective];
        if (role_canon != nullptr) {
          const std::uint32_t representative = role_canon[triple.connective];
          if (cue_scores[representative] > cue_weight)
            cue_weight = cue_scores[representative];
        }
        if (cue_weight < cue_identity_floor) cue_weight = 0u;
      }
      const bool cue_typed = cue_weight != 0u;
      std::uint32_t connective_order = kNoTripleUnit;
      if (cue_orders != nullptr) {
        connective_order = cue_orders[triple.connective];
        if (role_canon != nullptr) {
          const std::uint32_t representative = role_canon[triple.connective];
          if (cue_orders[representative] < connective_order)
            connective_order = cue_orders[representative];
        }
      }
      const bool operator_typed =
          cue_typed && asked_operator_order != kNoTripleUnit &&
          connective_order == asked_operator_order;
      const bool glue_typed = closed_class_mask[triple.connective] != 0u &&
          (triple.connective2 == kNoTripleUnit ||
           closed_class_mask[triple.connective2] != 0u);
      // An exact endpoint is already a device-grounded fact address.  It can
      // admit its learned content bridge even when a paraphrased question did
      // not repeat the observed predicate; otherwise A-P-[glue]-B facts can
      // be acquired but never queried.  Fuzzy/cohort candidates still need a
      // cue-typed or learned-glue bridge as before.
      const bool exact_endpoint =
          !topic_fallback && relation_cue_exact(role_canon, cue_exact, candidate_topic);
      if (!cue_typed && !glue_typed && !self_chain && !exact_endpoint) continue;
      std::uint32_t topic_weight = cue_scores[candidate_topic];
      std::uint32_t topic_order = cue_orders[candidate_topic];
      if (role_canon != nullptr) {
        const std::uint32_t representative = role_canon[candidate_topic];
        if (cue_scores[representative] > topic_weight)
          topic_weight = cue_scores[representative];
        if (cue_orders[representative] < topic_order)
          topic_order = cue_orders[representative];
      }
      const bool exact_topic = exact_endpoint;
      const bool direct_topic =
          !topic_fallback && exact_topic &&
          asked_operator_order != kNoTripleUnit &&
          topic_order > asked_operator_order;
      // The count tier gates UNTYPED candidates only: a single attested
      // claim in the ASKED relation outranks a repeated collocation in an
      // unasked one (the cue's type is itself the evidence).
      if (count < tier && !cue_typed && !analogical) continue;
      // Rank: the learned question-operator position first, then asked
      // relation identity, then ATTESTATION (count --
      // a repeated bridge is evidence, rarity is not; for an analogical
      // candidate the attestation is weighted by its category-mate
      // strength), then forward orientation (a claim whose subject IS the
      // topic), then the vitality-normalized strength as the final
      // discriminator.
      unsigned long long ranked_count = count;
      if (analogical && mate_counts != nullptr) {
        const unsigned long long shared = mate_counts[triple.subject];
        const unsigned long long degree =
            subject_degree != nullptr ? subject_degree[triple.subject] : 0ull;
        // Jaccard-style category confidence: shared contexts normalized by
        // the mate's total context degree. A hub subject overlaps everything
        // a little (large degree, small fraction) and is punished; a true
        // category mate shares a large FRACTION of its contexts. The value's
        // vitality divides as well, so generic filler values ("way",
        // "chance") lose to specific ones.
        const unsigned long long confidence = (shared << 10u) / (degree + 4ull);
        ranked_count =
            (static_cast<unsigned long long>(count) * confidence) /
            (static_cast<unsigned long long>(unit_vitality[other]) + 16ull);
      }
      const unsigned long long strength =
          (ranked_count << 12u) /
          (static_cast<unsigned long long>(unit_vitality[other]) + 16ull);
      bool better = false;
      if (operator_typed != best_operator_typed) {
        better = operator_typed;
      } else if (direct_topic != best_direct_topic) {
        better = direct_topic;
      } else if (exact_topic != best_exact_topic) {
        better = exact_topic;
      } else if (direct_topic && best_direct_topic &&
                 topic_order != best_topic_order) {
        // A learned question operator scopes the first exact content-bearing
        // endpoint after it. Closed-class material may intervene, so this is
        // an ordered stream relation rather than an immediate adjacency rule.
        better = topic_order < best_topic_order;
      } else {
        better = topic_weight > best_topic_weight;
        if (topic_weight == best_topic_weight) {
          if (topic_fallback && topic_order != best_topic_order) {
            better = topic_order > best_topic_order;
          } else {
            better = cue_weight > best_cue_weight;
            if (cue_weight == best_cue_weight) {
              better = ranked_count > best_count;
              if (ranked_count == best_count) {
                better = (forward && !best_forward) ||
                         (forward == best_forward &&
                          (strength > best_strength ||
                           (strength == best_strength &&
                            (best_end == kNoTripleUnit || other < best_end))));
              }
            }
          }
        }
      }
      if (!better) continue;
      best_topic_weight = topic_weight;
      best_topic_order = topic_order;
      best_cue_weight = cue_weight;
      best_strength = strength;
      best_count = ranked_count;
      best_forward = forward;
      best_operator_typed = operator_typed;
      best_direct_topic = direct_topic;
      best_exact_topic = exact_topic;
      best_exact_endpoint = exact_endpoint;
      best = entry;
      best_topic = candidate_topic;
      best_end = other;
    }
    if (best == kNoTripleUnit) {
      if (!self_chain && clauses == 0u && tier > 1u) {
        tier = 1u;  // admit single-attestation triples, surfaced in meta[3]
        continue;   // (first-sentence answer only; never for continuation)
      }
      break;
    }
    const bool best_analogical = (best & kAnalogyFlag) != 0u;
    const std::uint32_t slot = best & ~(0x80000000u | kAnalogyFlag);
    const RelationTriple triple = table[slot];
    const std::uint32_t plan_subject =
        best_analogical ? analogical_topic : triple.subject;
    // Recover at most two units of the subject's learned local phrase. Cue-
    // matched predecessors win; otherwise resident adjacency count decides.
    // This expands an argument, never the predicate or answer content, and
    // cannot copy an unbounded source span.
    std::uint32_t prefixes[2u] = {kNoTripleUnit, kNoTripleUnit};
    std::uint32_t prefix_count = 0u;
    std::uint32_t phrase_cursor = plan_subject;
    // Prefix recovery echoes the CUE's phrasing and belongs to the cued
    // first sentence; an analogical clause substitutes its subject, so a
    // cue-scored prefix would be echo noise ("is how talent...").
    for (std::uint32_t depth = 0u; !best_analogical && depth < 2u; ++depth) {
      std::uint32_t prefix = kNoTripleUnit;
      std::uint32_t prefix_cue = 0u;
      std::uint32_t prefix_count_mass = 0u;
      for (std::uint32_t edge = 0u; edge < phrase_bigram_count; ++edge) {
        if (phrase_bigrams[edge].next != phrase_cursor) continue;
        const std::uint32_t candidate = phrase_bigrams[edge].previous;
        if (candidate == phrase_cursor || candidate == triple.connective ||
            candidate == triple.value || unit_lengths[candidate] == 0u)
          continue;
        std::uint32_t candidate_cue = cue_scores[candidate];
        if (role_canon != nullptr) {
          const std::uint32_t representative = role_canon[candidate];
          if (cue_scores[representative] > candidate_cue)
            candidate_cue = cue_scores[representative];
        }
        if (candidate_cue < cue_identity_floor) candidate_cue = 0u;
        const std::uint32_t count_mass = phrase_bigram_counts[edge];
        if (candidate_cue > prefix_cue ||
            (candidate_cue == prefix_cue && count_mass > prefix_count_mass)) {
          prefix = candidate;
          prefix_cue = candidate_cue;
          prefix_count_mass = count_mass;
        }
      }
      if (prefix == kNoTripleUnit) break;
      if (depth == 0u &&
          (prefix_cue == 0u || closed_class_mask[prefix] != 0u))
        break;
      prefixes[prefix_count++] = prefix;
      phrase_cursor = prefix;
    }
    while (prefix_count != 0u && extent + prefix_count + 3u > kCommitmentCap)
      --prefix_count;
    for (std::uint32_t p = prefix_count; p != 0u; --p)
      plan[extent++] = prefixes[p - 1u];
    plan[extent++] = plan_subject;
    plan[extent++] = triple.connective;
    if (triple.connective2 != kNoTripleUnit) plan[extent++] = triple.connective2;
    plan[extent++] = triple.value;
    // Exact fact bridges may retain one observed successor of their selected
    // value. This is a bounded resident bigram continuation, not a copied
    // span: it keeps a learned multi-unit endpoint (for example a name plus
    // its learned qualifier) intact when the graph selected only its first
    // unit. No lexical class or answer label is supplied here.
    if (best_exact_endpoint && !self_chain && extent < kCommitmentCap) {
      std::uint32_t continuation = kNoTripleUnit;
      std::uint32_t continuation_count = 0u;
      for (std::uint32_t edge = 0u; edge < phrase_bigram_count; ++edge) {
        if (phrase_bigrams[edge].previous != triple.value) continue;
        const std::uint32_t candidate = phrase_bigrams[edge].next;
        if (candidate == triple.value || candidate >= unit_count ||
            closed_class_mask[candidate] != 0u ||
            !commitment_filler_legal(candidate, closed_class_mask, unit_lengths,
                                     unit_content, unit_words, boundary_mask,
                                     filler_terminal_mask))
          continue;
        bool used = false;
        for (std::uint32_t index = 0u; index < extent; ++index)
          used |= plan[index] == candidate;
        if (used) continue;
        const std::uint32_t count_mass = phrase_bigram_counts[edge];
        if (count_mass > continuation_count ||
            (count_mass == continuation_count && candidate < continuation)) {
          continuation = candidate;
          continuation_count = count_mass;
        }
      }
      if (continuation != kNoTripleUnit) plan[extent++] = continuation;
    }
    // ORDER-PRESERVING TAIL (self-chain fluency): extend the claim rightward
    // with the value's strongest learned continuation -- one glue unit plus
    // one content unit, committed ATOMICALLY (a dangling glue tail reads
    // broken). Append-only: the claim's own order can never scramble, unlike
    // skeleton slot binding (measured to shuffle it). Adjacency evidence
    // comes from the resident phrase-bigram store; nothing is authored.
    if (self_chain && extent + 2u <= kCommitmentCap) {
      std::uint32_t tail_glue = kNoTripleUnit;
      std::uint32_t tail_content = kNoTripleUnit;
      std::uint32_t cursor_unit = triple.value;
      for (std::uint32_t depth = 0u; depth < 2u; ++depth) {
        std::uint32_t next = kNoTripleUnit;
        std::uint32_t next_count = 0u;
        std::uint32_t next_noun = kNoTripleUnit;   // emergent-NOUN landing:
        std::uint32_t next_noun_count = 0u;        // a completed phrase ends
        for (std::uint32_t edge = 0u; edge < phrase_bigram_count; ++edge) {  // on substance
          if (phrase_bigrams[edge].previous != cursor_unit) continue;
          const std::uint32_t candidate = phrase_bigrams[edge].next;
          if (candidate == cursor_unit || unit_lengths[candidate] == 0u)
            continue;
          const std::uint32_t edge_count = phrase_bigram_counts[edge];
          const bool want_glue = depth == 0u;
          if (want_glue != (closed_class_mask[candidate] != 0u)) continue;
          bool used = false;
          for (std::uint32_t k = 0u; k < extent; ++k)
            used |= plan[k] == candidate;
          if (used) continue;
          if (!want_glue &&
              (unit_lengths[candidate] < kConstructionMinFillerBytes ||
               (unit_pos != nullptr && unit_pos[candidate] == 1u) ||
               commitment_near_duplicate(candidate, prior_ends,
                                         prior_end_count, unit_lengths,
                                         unit_content, unit_words)))
            continue;
          {
            bool artifact = false;
            for (std::uint32_t o = 0u;
                 o < unit_lengths[candidate] && !artifact; ++o) {
              const std::uint32_t b = construction_unit_byte(
                  unit_content, unit_words, candidate, o);
              artifact = b < 0x20u || b == 0x22u || b == 0x27u ||
                         b == 0x28u || b == 0x29u || b == 0x2eu ||
                         b == 0x5bu || b == 0x5du || b == 0x7bu ||
                         b == 0x7du || b == 0xe2u;
            }
            if (artifact) continue;
          }
          if (edge_count > next_count) {
            next = candidate;
            next_count = edge_count;
          }
          if (!want_glue && unit_pos != nullptr && unit_pos[candidate] == 2u &&
              edge_count > next_noun_count) {
            next_noun = candidate;
            next_noun_count = edge_count;
          }
        }
        // Prefer the strongest NOUN landing (the tail COMPLETES a phrase
        // instead of stalling on a modifier: "of economics", not "of
        // economic"); fall back to the strongest legal continuation.
        if (depth == 1u && next_noun != kNoTripleUnit && next_noun_count >= 2u) {
          next = next_noun;
          next_count = next_noun_count;
        }
        if (next == kNoTripleUnit || next_count < 2u) break;
        if (depth == 0u) tail_glue = next; else tail_content = next;
        cursor_unit = next;
      }
      if (tail_glue != kNoTripleUnit && tail_content != kNoTripleUnit) {
        plan[extent++] = tail_glue;
        plan[extent++] = tail_content;
      }
    }
    if (meta[2] == kNoTripleUnit) meta[2] = best_topic;
    if (clauses == 0u) meta[7] = slot;
    committed_ends[committed_end_count++] = best_topic;
    committed_ends[committed_end_count++] = best_end;
    used_connectives[clauses] = triple.connective;
    meta[4] = best_end;                      // this clause's rheme
    meta[5] = best_forward ? 1u : 0u;
    {
      std::uint32_t connective_canon = triple.connective;
      if (role_canon != nullptr && triple.connective < unit_count &&
          role_canon[triple.connective] < unit_count)
        connective_canon = role_canon[triple.connective];
      meta[6] = connective_canon;            // this clause's relation type
    }
    ++clauses;
  }
  meta[0] = extent;
  meta[1] = clauses;
  meta[3] = clauses != 0u ? tier : 0u;
}

// A relation table aggregates recurrence, but the surface organ may only
// realize a clause when the selected aggregate still has one exact learned
// construction witness. The revision is stamped by the shared extractor at
// acquisition time. Ambiguous or missing witnesses abstain instead of
// borrowing a construction from a neighboring relation.
__device__ inline void clear_relation_triple_plan_meta(std::uint32_t* meta) {
  meta[0] = 0u;
  meta[1] = 0u;
  meta[2] = kNoTripleUnit;
  meta[3] = 0u;
  meta[4] = kNoTripleUnit;
  meta[5] = 0u;
  meta[6] = kNoTripleUnit;
  meta[7] = kNoTripleUnit;
}

static __global__ void require_relation_construction_witness_kernel(
    const std::uint64_t* relation_evidence_revision,
    const std::uint64_t* construction_evidence_revision,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_count,
    std::uint32_t construction_capacity, std::uint32_t* meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || meta == nullptr) return;
  // The revision sidecar names one exact event. A multi-clause aggregate plan
  // has more than one source event and needs the retained path store; do not
  // silently validate only its first clause.
  if (meta[1] != 1u || relation_evidence_revision == nullptr ||
      construction_evidence_revision == nullptr || construction_lengths == nullptr ||
      construction_count == nullptr || meta[7] >= kRelationTripleHashCap) {
    clear_relation_triple_plan_meta(meta);
    return;
  }
  const std::uint64_t evidence = relation_evidence_revision[meta[7]];
  if (evidence == 0u) {
    clear_relation_triple_plan_meta(meta);
    return;
  }
  const std::uint32_t count =
      construction_count[0] < construction_capacity ? construction_count[0]
                                                   : construction_capacity;
  std::uint32_t matches = 0u;
  for (std::uint32_t index = 0u; index < count; ++index) {
    if (construction_lengths[index] == 0u ||
        construction_evidence_revision[index] != evidence)
      continue;
    ++matches;
    if (matches > 1u) {
      clear_relation_triple_plan_meta(meta);
      return;
    }
  }
  if (matches != 1u) clear_relation_triple_plan_meta(meta);
}

[[nodiscard]] __device__ inline bool witnessed_event_same_group(
    const WitnessedRelationEvent& left, const WitnessedRelationEvent& right) {
  return (left.evidence_revision >> 32u) ==
             (right.evidence_revision >> 32u) &&
         left.segment_begin == right.segment_begin;
}

[[nodiscard]] __device__ inline std::uint32_t witnessed_event_cue_hits(
    const WitnessedRelationEvent& event, const std::uint32_t* cue_exact,
    std::uint32_t unit_count) {
  const std::uint32_t units[4] = {
      event.triple.subject, event.triple.connective,
      event.triple.connective2, event.triple.value};
  std::uint32_t hits = 0u;
  for (std::uint32_t i = 0u; i < 4u; ++i) {
    const std::uint32_t unit = units[i];
    if (unit >= unit_count || unit == kNoTripleUnit) continue;
    bool duplicate = false;
    for (std::uint32_t j = 0u; j < i; ++j) duplicate |= units[j] == unit;
    if (!duplicate && cue_exact[unit] != 0u) ++hits;
  }
  return hits;
}

[[nodiscard]] __host__ __device__ inline bool
relation_surface_evidence_better(
    bool ordered_residual, std::uint32_t candidate_novel,
    std::uint32_t candidate_rank, std::uint64_t candidate_vitality,
    bool has_best, std::uint32_t best_novel, std::uint32_t best_rank,
    std::uint64_t best_vitality) {
  if (candidate_novel == 0u) return false;
  if (!has_best) return true;
  if (ordered_residual) {
    return candidate_rank < best_rank ||
           (candidate_rank == best_rank &&
            (candidate_novel > best_novel ||
             (candidate_novel == best_novel &&
              candidate_vitality < best_vitality)));
  }
  return candidate_novel > best_novel ||
         (candidate_novel == best_novel &&
          (candidate_rank < best_rank ||
           (candidate_rank == best_rank &&
            candidate_vitality < best_vitality)));
}

struct WitnessedRelationPlanReceipt {
  std::uint32_t requested_field = kRelationFieldCount;
  std::uint32_t best_hits = 0u;
  std::uint32_t best_event = kNoTripleUnit;
  std::uint32_t residual_count = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t best_requested_residual = 0u;
  std::uint32_t best_ambiguous = 0u;
  std::uint64_t best_revision = 0u;
  std::uint64_t next_revision = 0u;
  std::uint32_t best_connective_score = 0u;
  std::uint32_t next_connective_score = 0u;
  std::uint32_t next_event_score = 0u;
  std::uint32_t next_hits = 0u;
  std::uint32_t next_ambiguous = 0u;
  std::uint32_t topic_groups = 0u;
  std::uint32_t zero_residual_groups = 0u;
  std::uint32_t selected_next = 0u;
  std::uint32_t selected_topic = kNoTripleUnit;
  std::uint32_t scanned_live_events = 0u;
  std::uint32_t topic_events = 0u;
  std::uint32_t question_gap_ambiguous = 0u;
};

[[nodiscard]] __host__ __device__ inline bool
witnessed_evidence_follows(std::uint64_t previous_revision,
                           std::uint64_t candidate_revision) {
  return candidate_revision > previous_revision;
}

[[nodiscard]] __device__ inline std::uint32_t learned_question_gap_field(
    const std::uint32_t* cue_exact, const std::uint32_t* cue_orders,
    const std::uint64_t* qonset_evidence_revision,
    const std::uint32_t* question_gap_field_support,
    const std::uint32_t* role_canon, std::uint32_t unit_count,
    bool* ambiguous_out = nullptr) {
  if (ambiguous_out != nullptr) *ambiguous_out = false;
  if (cue_exact == nullptr || cue_orders == nullptr ||
      qonset_evidence_revision == nullptr ||
      question_gap_field_support == nullptr)
    return kRelationFieldCount;
  (void)role_canon;

  std::uint32_t opener = kNoTripleUnit;
  std::uint32_t opener_order = kNoTripleUnit;
  bool opener_ambiguous = false;
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
    if (cue_exact[unit] == 0u) continue;
    if (qonset_evidence_revision[unit] == 0u) continue;
    const std::uint32_t order = cue_orders[unit];
    if (opener == kNoTripleUnit || order < opener_order) {
      opener = unit;
      opener_order = order;
      opener_ambiguous = false;
    } else if (order == opener_order && unit != opener) {
      opener_ambiguous = true;
    }
  }
  if (opener == kNoTripleUnit || opener_ambiguous) {
    if (ambiguous_out != nullptr) *ambiguous_out = opener_ambiguous;
    return kRelationFieldCount;
  }

  std::uint32_t requested_field = kRelationFieldCount;
  std::uint32_t best_support = 0u;
  bool field_ambiguous = false;
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    const std::uint32_t support =
        question_gap_field_support[opener * kRelationFieldCount + field];
    if (support > best_support) {
      requested_field = field;
      best_support = support;
      field_ambiguous = false;
    } else if (support != 0u && support == best_support) {
      field_ambiguous = true;
    }
  }
  if (best_support == 0u || field_ambiguous)
    requested_field = kRelationFieldCount;
  if (ambiguous_out != nullptr) *ambiguous_out = field_ambiguous;
  return requested_field;
}

// Resolve the current cue against exact extracted relation events. Raw contact
// surfaces never participate, so a previous request cannot win by replaying
// its sentence. A sparsely grounded cue may traverse one uniquely strongest
// identity-sharing event group; ties abstain.
static __global__ void form_witnessed_relation_plan_kernel(
    const WitnessedRelationEvent* events, const std::uint32_t* event_cursor,
    const std::uint32_t* cue_exact, const std::uint32_t* cue_scores,
    const std::uint32_t* cue_orders,
    const std::uint64_t* qonset_evidence_revision,
    const std::uint32_t* qonset_count,
    // Resident relation store (same table learn_relation_triples_kernel
    // fills online). requested_field below is now derived LIVE from this,
    // per query, via the causal-compatibility vote -- not from the retired
    // static per-opener question_gap_field_support table, which the old
    // learn_question_gap_fields_kernel writer never actually populated
    // (differences==1u fired zero times on real discourse; see 0X1-156).
    const RelationTriple* relation_triple_table,
    const std::uint32_t* relation_triple_table_counts,
    const std::uint32_t* role_canon,
    const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_vitality,
    const std::uint32_t* event_surface_units,
    const std::uint32_t* event_surface_counts,
    std::uint32_t unit_count, bool topic_fallback,
    std::uint32_t* plan, std::uint32_t* meta,
    std::uint64_t* selected_evidence_revisions,
    std::uint32_t selected_evidence_capacity,
    std::uint32_t* selected_evidence_count,
    WitnessedRelationPlanReceipt* receipt = nullptr,
    const std::uint64_t* selected_topic_key = nullptr) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || events == nullptr ||
      event_cursor == nullptr || cue_exact == nullptr ||
      closed_class_mask == nullptr || event_surface_units == nullptr ||
      event_surface_counts == nullptr || plan == nullptr || meta == nullptr)
    return;
  if (selected_evidence_count != nullptr) selected_evidence_count[0] = 0u;
  if (receipt != nullptr) receipt[0] = WitnessedRelationPlanReceipt{};
  (void)qonset_count;
  (void)cue_orders;
  (void)role_canon;
  if (selected_evidence_revisions != nullptr)
    for (std::uint32_t index = 0u; index < selected_evidence_capacity; ++index)
      selected_evidence_revisions[index] = 0u;
  for (std::uint32_t i = 0u; i < kCommitmentCap; ++i)
    plan[i] = kNoTripleUnit;
  clear_relation_triple_plan_meta(meta);
  const std::uint32_t extent =
      event_cursor[0] < kWitnessedRelationEventCap
          ? event_cursor[0]
          : kWitnessedRelationEventCap;
  const std::uint32_t first =
      event_cursor[0] > kWitnessedRelationEventCap
          ? event_cursor[0] & (kWitnessedRelationEventCap - 1u)
          : 0u;
  // requested_field cannot be derived yet: the causal-compatibility vote
  // below needs the winning witnessed event's own triple, which the scan
  // below has not found yet. It stays the sentinel through best-selection
  // (bit-identical to every prior run, since the retired static reader this
  // replaced was itself permanently sentinel there) and is recomputed live,
  // from the resident relation store, immediately after `best` is known.
  std::uint32_t requested_field = kRelationFieldCount;
  bool question_gap_ambiguous = false;
  std::uint32_t best = kNoTripleUnit;
  std::uint32_t best_hits = 0u;
  std::uint32_t best_score = 0u;
  std::uint32_t best_requested_residual = 0u;
  bool best_ambiguous = false;
  std::uint64_t best_revision = 0u;
  std::uint32_t contact_best = kNoTripleUnit;
  std::uint32_t contact_best_hits = 0u;
  std::uint32_t contact_best_score = 0u;
  std::uint32_t contact_hits = 0u;
  std::uint32_t contact_hit_units[kCommitmentCap];
  std::uint32_t contact_hit_count = 0u;
  std::uint32_t contact_residual_units[kCommitmentCap];
  std::uint32_t contact_residual_count = 0u;
  std::uint32_t contact_requested_residual_units[kCommitmentCap];
  std::uint32_t contact_requested_residual_count = 0u;
  std::uint64_t contact_revision = 0u;
  std::uint32_t contact_segment_begin = kNoTripleUnit;
  const std::uint32_t selected_topic =
      selected_topic_key == nullptr || selected_topic_key[0] == 0u
          ? kNoTripleUnit
          : 0xffffu -
                static_cast<std::uint32_t>(
                    selected_topic_key[0] & 0xffffu);
  const std::uint32_t selected_topic_revision =
      selected_topic_key == nullptr
          ? 0u
          : static_cast<std::uint32_t>(
                (selected_topic_key[0] >> 32u) & 0x0fffffffu);
  if (receipt != nullptr) receipt->selected_topic = selected_topic;
  bool contact_has_topic = false;
  for (std::uint32_t offset = 0u; offset <= extent; ++offset) {
    const std::uint32_t i =
        (first + offset) & (kWitnessedRelationEventCap - 1u);
    const bool at_end = offset == extent;
    const WitnessedRelationEvent event =
        at_end ? WitnessedRelationEvent{} : events[i];
    const std::uint64_t revision =
        at_end ? 0u : event.evidence_revision >> 32u;
    if ((at_end ||
         (contact_revision != 0u &&
          (revision != contact_revision ||
           event.segment_begin != contact_segment_begin))) &&
        contact_revision != 0u) {
      const bool has_contact_best = contact_best != kNoTripleUnit;
      const std::uint32_t contact_residual =
          has_contact_best ? contact_residual_count : 0u;
      const std::uint32_t contact_requested_residual =
          requested_field < kRelationFieldCount
              ? (has_contact_best ? contact_requested_residual_count : 0u)
              : contact_residual;
      const bool topic_admitted =
          selected_topic == kNoTripleUnit || contact_has_topic ||
          (selected_topic_revision != 0u &&
           contact_revision == selected_topic_revision && contact_hits != 0u);
      if (topic_admitted) {
        if (receipt != nullptr) ++receipt->topic_groups;
        if (contact_residual == 0u && receipt != nullptr)
          ++receipt->zero_residual_groups;
      }
      const bool better =
          topic_admitted && contact_residual != 0u &&
          (best == kNoTripleUnit || contact_hits > best_hits ||
           (contact_hits == best_hits &&
            (contact_best_score > best_score ||
             (contact_best_score == best_score &&
              (contact_requested_residual > best_requested_residual ||
               (contact_requested_residual == best_requested_residual &&
                contact_revision > best_revision))))));
      const bool exact_same_revision_tie =
          topic_admitted && contact_residual != 0u &&
          best != kNoTripleUnit && contact_hits == best_hits &&
          contact_best_score == best_score &&
          contact_requested_residual == best_requested_residual &&
          contact_revision == best_revision &&
          contact_segment_begin != events[best].segment_begin;
      if (better) {
        best = contact_best;
        best_hits = contact_hits;
        best_score = contact_best_score;
        best_requested_residual = contact_requested_residual;
        best_revision = contact_revision;
        best_ambiguous = false;
      } else if (exact_same_revision_tie) {
        best_ambiguous = true;
      }
      contact_best = kNoTripleUnit;
      contact_best_hits = 0u;
      contact_best_score = 0u;
      contact_hits = 0u;
      contact_hit_count = 0u;
      contact_residual_count = 0u;
      contact_requested_residual_count = 0u;
      contact_revision = 0u;
      contact_segment_begin = kNoTripleUnit;
      contact_has_topic = false;
    }
    if (at_end) break;
    // Request modality is fixed at acquisition. Every retained event passed
    // the write-side contact modality gate; a later onset update must not
    // retroactively erase it.
    if (event.live == 0u || event.evidence_revision == 0u)
      continue;
    if (receipt != nullptr) ++receipt->scanned_live_events;
    if (contact_revision == 0u) {
      contact_revision = revision;
      contact_segment_begin = event.segment_begin;
    }
    const std::uint32_t hits =
        witnessed_event_cue_hits(event, cue_exact, unit_count);
    const std::uint32_t units[4] = {
        event.triple.subject, event.triple.connective,
        event.triple.connective2, event.triple.value};
    const std::uint32_t residual_units[5] = {
        event.triple.subject, event.triple.connective,
        event.triple.connective2, event.triple.value, event.terminal};
    for (std::uint32_t field = 0u;
         field < 5u && contact_residual_count < kCommitmentCap; ++field) {
      const std::uint32_t unit = residual_units[field];
      if (unit >= unit_count || unit == kNoTripleUnit ||
          closed_class_mask[unit] != 0u || cue_exact[unit] != 0u)
        continue;
      bool seen = false;
      for (std::uint32_t prior = 0u; prior < contact_residual_count; ++prior)
        seen |= contact_residual_units[prior] == unit;
      if (!seen) contact_residual_units[contact_residual_count++] = unit;
    }
    if (requested_field < kRelationFieldCount &&
        contact_requested_residual_count < kCommitmentCap) {
      const std::uint32_t unit = residual_units[requested_field];
      if (unit < unit_count && unit != kNoTripleUnit &&
          closed_class_mask[unit] == 0u && cue_exact[unit] == 0u) {
        bool seen = false;
        for (std::uint32_t prior = 0u;
             prior < contact_requested_residual_count; ++prior)
          seen |= contact_requested_residual_units[prior] == unit;
        if (!seen)
          contact_requested_residual_units[
              contact_requested_residual_count++] = unit;
      }
    }
    const bool event_has_topic =
        selected_topic != kNoTripleUnit &&
        (event.triple.subject == selected_topic ||
         event.triple.value == selected_topic);
    contact_has_topic |= event_has_topic;
    if (event_has_topic && receipt != nullptr) ++receipt->topic_events;
    for (std::uint32_t u = 0u;
         u < 4u && contact_hit_count < kCommitmentCap; ++u) {
      const std::uint32_t unit = units[u];
      if (unit >= unit_count || unit == kNoTripleUnit) continue;
      if (cue_exact[unit] == 0u) continue;
      bool seen = false;
      for (std::uint32_t h = 0u; h < contact_hit_count; ++h)
        seen |= contact_hit_units[h] == unit;
      if (!seen) contact_hit_units[contact_hit_count++] = unit;
    }
    contact_hits = contact_hit_count;
    std::uint32_t event_score = 0u;
    if (cue_scores != nullptr) {
      for (std::uint32_t u = 0u; u < 4u; ++u) {
        const std::uint32_t unit = units[u];
        if (unit < unit_count && unit != kNoTripleUnit &&
            closed_class_mask[unit] == 0u)
          event_score += cue_scores[unit];
      }
    }
    if (hits > contact_best_hits ||
        (hits == contact_best_hits && event_score > contact_best_score)) {
      contact_best = i;
      contact_best_hits = hits;
      contact_best_score = event_score;
    }
  }
  if (receipt != nullptr) {
    receipt->best_hits = best_hits;
    receipt->best_event = best;
    receipt->best_requested_residual = best_requested_residual;
    receipt->best_ambiguous = best_ambiguous ? 1u : 0u;
    receipt->best_revision = best_revision;
  }
  if (best == kNoTripleUnit || best_hits == 0u || best_ambiguous) return;

  // LIVE requested_field, wired 2026-08-14 (0X1-156): now that `best` names
  // the actual witnessed relation this cue resolved to, vote on which of its
  // fields the question left as a gap -- causal-compatibility support from
  // the resident store, not the retired static per-opener table.
  requested_field = requested_field_from_causal_compatibility(
      relation_triple_table, relation_triple_table_counts, events[best].triple,
      cue_exact, closed_class_mask, unit_count, &question_gap_ambiguous);
  if (receipt != nullptr) {
    receipt->requested_field = requested_field;
    receipt->question_gap_ambiguous = question_gap_ambiguous ? 1u : 0u;
  }

  std::uint32_t residual[kCommitmentCap];
  std::uint32_t residual_count = 0u;
  const std::uint32_t passes =
      requested_field < kRelationFieldCount ? 2u : 1u;
  for (std::uint32_t pass = 0u; pass < passes; ++pass) {
    for (std::uint32_t offset = 0u; offset < extent; ++offset) {
      const std::uint32_t i =
          (first + offset) & (kWitnessedRelationEventCap - 1u);
      if (events[i].live == 0u ||
          !witnessed_event_same_group(events[best], events[i]))
        continue;
      const std::uint32_t units[5] = {
          events[i].triple.subject, events[i].triple.connective,
          events[i].triple.connective2, events[i].triple.value,
          events[i].terminal};
      for (std::uint32_t j = 0u;
           j < 5u && residual_count < kCommitmentCap; ++j) {
        if (requested_field < kRelationFieldCount &&
            ((pass == 0u) != (j == requested_field)))
          continue;
        const std::uint32_t unit = units[j];
        if (unit >= unit_count || unit == kNoTripleUnit ||
            closed_class_mask[unit] != 0u || cue_exact[unit] != 0u)
          continue;
        bool seen = false;
        for (std::uint32_t r = 0u; r < residual_count; ++r)
          seen |= residual[r] == unit;
        if (!seen) residual[residual_count++] = unit;
      }
    }
  }
  if (receipt != nullptr) receipt->residual_count = residual_count;
  if (residual_count == 0u) return;
  std::uint32_t best_connective_score = 0u;
  if (cue_scores != nullptr) {
    for (std::uint32_t offset = 0u; offset < extent; ++offset) {
      const std::uint32_t i =
          (first + offset) & (kWitnessedRelationEventCap - 1u);
      if (events[i].live == 0u ||
          (events[i].evidence_revision >> 32u) != best_revision)
        continue;
      const std::uint32_t connectives[2] = {
          events[i].triple.connective, events[i].triple.connective2};
      for (std::uint32_t c = 0u; c < 2u; ++c) {
        const std::uint32_t unit = connectives[c];
        if (unit < unit_count && unit != kNoTripleUnit &&
            closed_class_mask[unit] == 0u &&
            cue_scores[unit] > best_connective_score)
          best_connective_score = cue_scores[unit];
      }
    }
  }

  // A cue with only one supported identity names a starting point rather than
  // a complete fact. Follow one unique event group sharing the strongest
  // residual identity mass and expose only that group's novel endpoint matter.
  std::uint32_t next = kNoTripleUnit;
  std::uint32_t next_hits = 0u;
  std::uint32_t next_connective_score = 0u;
  std::uint32_t next_event_score = 0u;
  std::uint64_t next_revision = 0u;
  std::uint32_t next_segment_begin = kNoTripleUnit;
  bool next_ambiguous = false;
  {
    contact_best = kNoTripleUnit;
    contact_best_hits = 0u;
    contact_best_score = 0u;
    contact_hits = 0u;
    contact_hit_count = 0u;
    std::uint32_t contact_connective_score = 0u;
    contact_revision = 0u;
    contact_segment_begin = kNoTripleUnit;
    for (std::uint32_t offset = 0u; offset <= extent; ++offset) {
      const std::uint32_t i =
          (first + offset) & (kWitnessedRelationEventCap - 1u);
      const bool at_end = offset == extent;
      const WitnessedRelationEvent event =
          at_end ? WitnessedRelationEvent{} : events[i];
      const std::uint64_t revision =
          at_end ? 0u : event.evidence_revision >> 32u;
      if ((at_end ||
           (contact_revision != 0u &&
            (revision != contact_revision ||
             event.segment_begin != contact_segment_begin))) &&
          contact_revision != 0u) {
        if (contact_hits > next_hits ||
            (contact_hits == next_hits &&
             (contact_connective_score > next_connective_score ||
              (contact_connective_score == next_connective_score &&
               (contact_best_score > next_event_score ||
                (contact_best_score == next_event_score &&
                 next_revision != 0u &&
                 contact_revision < next_revision)))))) {
          next = contact_best;
          next_hits = contact_hits;
          next_connective_score = contact_connective_score;
          next_event_score = contact_best_score;
          next_revision = contact_revision;
          next_segment_begin = contact_segment_begin;
          next_ambiguous = false;
        } else if (contact_hits == next_hits && contact_hits != 0u &&
                   contact_connective_score == next_connective_score &&
                   contact_best_score == next_event_score &&
                   contact_revision == next_revision &&
                   contact_segment_begin != next_segment_begin) {
          next_ambiguous = true;
        }
        contact_best = kNoTripleUnit;
        contact_best_hits = 0u;
        contact_best_score = 0u;
        contact_hits = 0u;
        contact_hit_count = 0u;
        contact_connective_score = 0u;
        contact_revision = 0u;
        contact_segment_begin = kNoTripleUnit;
      }
      if (at_end) break;
      if (event.live == 0u || event.evidence_revision == 0u ||
          revision <= best_revision ||
          witnessed_event_same_group(events[best], event))
        continue;
      if (contact_revision == 0u) {
        contact_revision = revision;
        contact_segment_begin = event.segment_begin;
      }
      std::uint32_t hits = 0u;
      const std::uint32_t units[2] = {
          event.triple.subject, event.triple.value};
      for (std::uint32_t u = 0u; u < 2u; ++u)
        for (std::uint32_t r = 0u; r < residual_count; ++r)
          hits += units[u] == residual[r];
      for (std::uint32_t u = 0u;
           u < 2u && contact_hit_count < kCommitmentCap; ++u) {
        const std::uint32_t unit = units[u];
        bool shared = false;
        for (std::uint32_t r = 0u; r < residual_count; ++r)
          shared |= residual[r] == unit;
        if (!shared) continue;
        bool seen = false;
        for (std::uint32_t h = 0u; h < contact_hit_count; ++h)
          seen |= contact_hit_units[h] == unit;
        if (!seen) contact_hit_units[contact_hit_count++] = unit;
      }
      contact_hits = contact_hit_count;
      const std::uint32_t cue_units[2] = {
          event.triple.connective, event.triple.connective2};
      for (std::uint32_t u = 0u; u < 2u; ++u) {
        const std::uint32_t unit = cue_units[u];
        if (unit < unit_count && unit != kNoTripleUnit &&
            closed_class_mask[unit] == 0u &&
            cue_scores != nullptr &&
            cue_scores[unit] > contact_connective_score)
          contact_connective_score = cue_scores[unit];
      }
      std::uint32_t event_score = 0u;
      if (cue_scores != nullptr) {
        const std::uint32_t event_units[4] = {
            event.triple.subject, event.triple.connective,
            event.triple.connective2, event.triple.value};
        for (std::uint32_t u = 0u; u < 4u; ++u) {
          const std::uint32_t unit = event_units[u];
          if (unit < unit_count && unit != kNoTripleUnit &&
              closed_class_mask[unit] == 0u)
            event_score += cue_scores[unit];
        }
      }
      if (hits > contact_best_hits ||
          (hits == contact_best_hits && event_score > contact_best_score)) {
        contact_best = i;
        contact_best_hits = hits;
        contact_best_score = event_score;
      }
    }
  }
  if (receipt != nullptr) receipt->next_revision = next_revision;
  if (receipt != nullptr) {
    receipt->best_connective_score = best_connective_score;
    receipt->next_connective_score = next_connective_score;
    receipt->next_event_score = next_event_score;
    receipt->next_hits = next_hits;
    receipt->next_ambiguous = next_ambiguous ? 1u : 0u;
  }

  std::uint32_t extent_out = 0u;
  if (next != kNoTripleUnit && next_hits != 0u &&
      (next_hits > best_hits || next_connective_score != 0u ||
       next_event_score != 0u) &&
      !next_ambiguous) {
    for (std::uint32_t at = extent; at != 0u; --at) {
      const std::uint32_t i =
          (first + at - 1u) & (kWitnessedRelationEventCap - 1u);
      if (events[i].live == 0u ||
          !witnessed_event_same_group(events[next], events[i]))
        continue;
      const std::uint32_t units[5] = {
          events[i].triple.subject, events[i].triple.connective,
          events[i].triple.connective2, events[i].triple.value,
          events[i].terminal};
      for (std::uint32_t at_unit = 5u;
           at_unit != 0u && extent_out < kCommitmentCap; --at_unit) {
        const std::uint32_t unit = units[at_unit - 1u];
        if (unit >= unit_count || unit == kNoTripleUnit ||
            closed_class_mask[unit] != 0u || cue_exact[unit] != 0u)
          continue;
        bool shared = false;
        for (std::uint32_t r = 0u; r < residual_count; ++r)
          shared |= residual[r] == unit;
        bool seen = false;
        for (std::uint32_t r = 0u; r < extent_out; ++r)
          seen |= plan[r] == unit;
        if (!shared && !seen) plan[extent_out++] = unit;
      }
    }
  } else {
    // Episode selection already resolves cross-contact recency. Preserve the
    // observed order inside that episode so an earlier requested relation
    // (for example, "closed") is not demoted behind a later relation from the
    // same contact ("moved") before surface evidence is ranked.
    for (std::uint32_t i = 0u; i < residual_count; ++i)
      plan[extent_out++] = residual[i];
  }
  if (receipt != nullptr) receipt->output_count = extent_out;
  if (extent_out == 0u) return;
  const std::uint32_t selected_event =
      next != kNoTripleUnit && next_hits != 0u &&
              (next_hits > best_hits || next_connective_score != 0u ||
               next_event_score != 0u) &&
              !next_ambiguous
          ? next
          : best;
  if (receipt != nullptr) receipt->selected_next = selected_event == next ? 1u : 0u;
  const bool ordered_residual =
      requested_field < kRelationFieldCount ||
      (topic_fallback && requested_field >= kRelationFieldCount) ||
      selected_event == next;
  meta[0] = extent_out;
  meta[1] = 1u;
  meta[2] = events[best].triple.subject;
  meta[3] = 1u;
  meta[4] = plan[extent_out - 1u];
  meta[5] = 1u;
  meta[6] = kNoTripleUnit;
  meta[7] = 0u;
  if (selected_evidence_revisions != nullptr &&
      selected_evidence_count != nullptr && selected_evidence_capacity != 0u) {
    std::uint32_t selected_indices[kRelationSurfaceEvidenceCap];
    std::uint32_t selected_count = 0u;
    std::uint32_t covered_plan_mask = 0u;
    const std::uint32_t limit =
        min(selected_evidence_capacity, kRelationSurfaceEvidenceCap);
    while (selected_count < limit) {
      std::uint32_t best_index = kNoTripleUnit;
      std::uint32_t best_novel = 0u;
      std::uint32_t best_rank = kCommitmentCap;
      std::uint32_t best_mask = 0u;
      std::uint64_t best_vitality = UINT64_MAX;
      for (std::uint32_t offset = 0u; offset < extent; ++offset) {
        const std::uint32_t index =
            (first + offset) & (kWitnessedRelationEventCap - 1u);
        if (events[index].live == 0u ||
            !witnessed_event_same_group(events[selected_event], events[index]))
          continue;
        bool already_selected = false;
        for (std::uint32_t prior = 0u; prior < selected_count; ++prior)
          already_selected |= selected_indices[prior] == index;
        if (already_selected) continue;
        // A retained episode may expose many overlapping relation views.
        // Coverage chooses the first useful view; continuation must then move
        // forward through the acquisition revision. Revisiting an earlier
        // view turns one episode into an unordered quote bag.
        if (ordered_residual && selected_count != 0u &&
            !witnessed_evidence_follows(
                events[selected_indices[selected_count - 1u]]
                    .evidence_revision,
                events[index].evidence_revision))
          continue;
        const std::uint32_t surface_count = event_surface_counts[index];
        if (surface_count == 0u ||
            surface_count > kConstructionMaxTokens)
          continue;
        std::uint32_t event_mask = 0u;
        std::uint32_t rank = kCommitmentCap;
        for (std::uint32_t plan_index = 0u; plan_index < extent_out;
             ++plan_index) {
          bool matches = false;
          for (std::uint32_t unit_index = 0u;
               unit_index < surface_count; ++unit_index)
            matches |= event_surface_units[
                           index * kConstructionMaxTokens + unit_index] ==
                       plan[plan_index];
          if (!matches) continue;
          event_mask |= 1u << plan_index;
          if (plan_index < rank) rank = plan_index;
        }
        const std::uint32_t novel =
            __popc(event_mask & ~covered_plan_mask);
        std::uint64_t vitality = 0u;
        for (std::uint32_t plan_index = 0u; plan_index < extent_out;
             ++plan_index) {
          if ((event_mask & ~covered_plan_mask &
               (1u << plan_index)) == 0u)
            continue;
          vitality += unit_vitality == nullptr
                          ? 0u
                          : unit_vitality[plan[plan_index]];
        }
        // A traversed group already has its learned endpoint at the front of
        // the residual plan, so retain that ordering. A direct episode has no
        // such cross-event ordering authority; there the densest residual
        // witness must win before its local position. Both modes remain
        // resident evidence tournaments and use vitality only to break an
        // otherwise exact structural tie.
        const bool better = relation_surface_evidence_better(
            ordered_residual, novel, rank, vitality,
            best_index != kNoTripleUnit, best_novel, best_rank, best_vitality);
        if (better) {
          best_index = index;
          best_novel = novel;
          best_rank = rank;
          best_mask = event_mask;
          best_vitality = vitality;
        }
      }
      if (best_index == kNoTripleUnit || best_novel == 0u) break;
      selected_indices[selected_count++] = best_index;
      covered_plan_mask |= best_mask;
    }
    // The coverage winner for a traversed episode is often the terminal
    // compound frame. Recover at most one exact local predecessor whose
    // observed suffix is that frame's prefix. Reject predecessors that begin
    // in already-satisfied query/residual matter: those would replay the
    // cross-contact bridge instead of completing the learned local clause.
    if (ordered_residual && selected_event == next && selected_count != 0u &&
        selected_count < limit) {
      const std::uint32_t first_selected = selected_indices[0];
      const std::uint32_t selected_surface_count =
          event_surface_counts[first_selected];
      const std::uint32_t selected_prefix =
          selected_surface_count == 0u
              ? kNoTripleUnit
              : event_surface_units[first_selected * kConstructionMaxTokens];
      std::uint32_t predecessor = kNoTripleUnit;
      std::uint32_t predecessor_mask = 0u;
      std::uint32_t predecessor_novel = 0u;
      std::uint64_t predecessor_revision = 0u;
      for (std::uint32_t offset = 0u; offset < extent; ++offset) {
        const std::uint32_t index =
            (first + offset) & (kWitnessedRelationEventCap - 1u);
        if (events[index].live == 0u ||
            !witnessed_event_same_group(events[first_selected], events[index]) ||
            events[index].evidence_revision >=
                events[first_selected].evidence_revision)
          continue;
        const std::uint32_t surface_count = event_surface_counts[index];
        if (surface_count == 0u || surface_count > kConstructionMaxTokens)
          continue;
        const std::uint32_t surface_begin =
            event_surface_units[index * kConstructionMaxTokens];
        const std::uint32_t surface_end =
            event_surface_units[index * kConstructionMaxTokens +
                                surface_count - 1u];
        if (surface_end != selected_prefix || surface_begin >= unit_count ||
            cue_exact[surface_begin] != 0u)
          continue;
        bool starts_in_residual = false;
        for (std::uint32_t residual_index = 0u;
             residual_index < residual_count; ++residual_index)
          starts_in_residual |= surface_begin == residual[residual_index];
        if (starts_in_residual) continue;
        std::uint32_t event_mask = 0u;
        for (std::uint32_t plan_index = 0u; plan_index < extent_out;
             ++plan_index)
          for (std::uint32_t unit_index = 0u; unit_index < surface_count;
               ++unit_index)
            if (event_surface_units[
                    index * kConstructionMaxTokens + unit_index] ==
                plan[plan_index])
              event_mask |= 1u << plan_index;
        const std::uint32_t novel =
            __popc(event_mask & ~covered_plan_mask);
        if (novel == 0u) continue;
        if (predecessor == kNoTripleUnit || novel > predecessor_novel ||
            (novel == predecessor_novel &&
             events[index].evidence_revision > predecessor_revision)) {
          predecessor = index;
          predecessor_mask = event_mask;
          predecessor_novel = novel;
          predecessor_revision = events[index].evidence_revision;
        }
      }
      if (predecessor != kNoTripleUnit) {
        for (std::uint32_t index = selected_count; index != 0u; --index)
          selected_indices[index] = selected_indices[index - 1u];
        selected_indices[0] = predecessor;
        ++selected_count;
        covered_plan_mask |= predecessor_mask;
      }
    }
    // Keep the residual-coverage tournament order. Re-sorting by source
    // chronology makes an earlier pairwise view outrank the more complete
    // compound event that covered the same plan mass, recreating replay
    // fragments after the device has already selected better evidence.
    for (std::uint32_t index = 0u; index < selected_count; ++index)
      selected_evidence_revisions[index] =
          events[selected_indices[index]].evidence_revision;
    selected_evidence_count[0] = selected_count;
  }
}

#include "bcc32_cuda_resident_construction_composer_generation_tail.inl"

}  // namespace substrate::bcc32::resident_construction
