// Adult conditioned surface and question-origination learning kernels.
//
// Included after conditioned relation bindings and before generation.
// This unit owns cue-derived subject/question surface learning and
// deterministic resident morphology helpers; it adds no state.

__device__ std::uint32_t choose_weighted(std::uint32_t* rng,
                                        const std::uint32_t* ids,
                                        const std::uint32_t* weights,
                                        std::uint32_t count) {
  std::uint32_t total = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) total += weights[i];
  std::uint32_t x = *rng;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *rng = x;
  if (total == 0u) return ids[0];
  std::uint32_t draw = x % total;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (draw < weights[i]) return ids[i];
    draw -= weights[i];
  }
  return ids[count - 1u];
}
__device__ std::uint32_t choose_unigram(const std::uint32_t* top_ids,
                                       const std::uint32_t* vitality,
                                       std::uint32_t* rng) {
  std::uint32_t weights[kTopK];
  for (std::uint32_t i = 0u; i < kTopK; ++i) weights[i] = vitality[top_ids[i]];
  return choose_weighted(rng, top_ids, weights, kTopK);
}


// Collect the cue's content units into the resident subject field: every unit
// that near-identity matched a cue segment (cue_scores >= kCueNearIdentity),
// is long enough to carry content (>= kSubjectMinLen bytes), and is NOT an
// emergent FUNC-class unit (unit_pos == 1, e.g. "how"/"should"/"what"/"and"
// from the question phrasing) is added with a length-weighted initial
// activation. Excluding FUNC units matters: without it, generic cue words are
// as strong a subject anchor as the actual topic word, so drift re-anchoring
// competes between the real topic and connective words and the answer keeps
// re-latching onto "how"/"should" instead of the question's subject. This is
// the resident, online, cue-derived topical field the generator holds active
// across the whole continuation. unit_pos may be null (not yet built): treat
// that as "no FUNC info available" and fall back to the length-only filter.
// Host-side, content-ordered replacement for the retired GPU kernel version
// of this collection step. The GPU version assigned array slots via
// `atomicAdd(subject_count, 1u)` -- first-come-first-served by thread/warp
// scheduling order, not content. Confirmed nondeterministic by direct
// instrumentation (2026-07-21): identical repeated invocations on the same
// corpus/cue produced a stable subject_count (order-independent sum) but a
// DIFFERENT subject_ids/subject_weights array in 6 of 8 runs (FNV hash of
// the arrays), and a downstream positional-argmax weight-tie-break
// (generate_kernel's reinforce path) resolves ties by array index, so the
// schedule-dependent order propagated into a schedule-dependent generated
// answer despite generate_kernel's own scoring math being deterministic.
// This collects in a fixed, content-derived order (ascending unit id) on the
// host instead of GPU thread-arrival order, so repeated runs on identical
// state always produce the identical subject field (and therefore a
// deterministic downstream tie-break, since it now operates on a
// deterministic array). unit_count is bounded (~10^4-10^5 for real corpora)
// so a host-side pass here, once per cue (not per generation step), costs
// microseconds -- the same one-time host-copy trade build_unit_pos already
// makes safely in this file.
// QUESTION-ORIGINATION learning (--qorig). A thread fires only where the
// ONLINE stream itself placed a '?'-terminated unit -- a discovered corpus
// event (resident_unit_ends_with already used elsewhere in this file for
// closure/terminator checks), never an authored rule. It scans back through
// online_episode_units (unit granularity, not raw bytes -- BCC-32 already has
// discovered unit boundaries, unlike the byte-level scan the reference
// mechanism used) to the previous strong sentence terminator or episode
// start, then records two roles: the '?'-fused unit itself into
// qterm_count, and the farthest-back unit in the span into qonset_count. No
// unit is privileged -- interrogative-form onsets (what/how/where...)
// accumulate high onset counts only because they recur as the first unit
// across many differently-topiced questions; a content word occurs as an
// onset in almost none, so its count stays near zero (this is what the
// specificity gate in inject_qorig_units_deterministic below exploits).
__global__ void learn_qonset_terminal_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* online_episode_units,
    std::uint32_t online_episode_begin, std::uint32_t online_episode_count,
    const std::uint32_t* online_episode_breaks, std::uint32_t online_episode_break_count,
    const std::uint32_t* role_canon,
    const std::uint64_t* contact_evidence_revision,
    std::uint64_t* qonset_evidence_revision,
    std::uint32_t* qonset_count, std::uint32_t* qterm_count,
    std::uint32_t unit_count) {
  const std::uint32_t i =
      online_episode_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= online_episode_count) return;
  const std::uint32_t unit = online_episode_units[i];
  const std::uint32_t space = static_cast<std::uint32_t>(' ');
  const bool interrogative =
      resident_unit_terminal_byte(unit_lengths, unit_content, unit,
                                  static_cast<std::uint32_t>('?'), space);
  const bool declarative_terminal =
      resident_unit_terminal_byte(unit_lengths, unit_content, unit,
                                  static_cast<std::uint32_t>('.'), space) ||
      resident_unit_terminal_byte(unit_lengths, unit_content, unit,
                                  static_cast<std::uint32_t>('!'), space) ||
      resident_unit_terminal_byte(unit_lengths, unit_content, unit,
                                  static_cast<std::uint32_t>('\n'), space);
  if (!interrogative && !declarative_terminal) return;
  std::uint32_t lo = 0u;
  std::uint32_t hi = online_episode_break_count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (online_episode_breaks[mid] <= i) lo = mid + 1u; else hi = mid;
  }
  const std::uint32_t episode_begin = lo == 0u ? 0u : online_episode_breaks[lo - 1u];
  const std::uint32_t scan_floor =
      (i > episode_begin + kQOrigMaxSpan) ? i - kQOrigMaxSpan : episode_begin;
  std::uint32_t onset_position = i;
  for (std::uint32_t at = i; at > scan_floor; --at) {
    const std::uint32_t prev = online_episode_units[at - 1u];
    if (resident_unit_terminal_byte(unit_lengths, unit_content, prev, static_cast<std::uint32_t>('.'), space) ||
        resident_unit_terminal_byte(unit_lengths, unit_content, prev, static_cast<std::uint32_t>('!'), space) ||
        resident_unit_terminal_byte(unit_lengths, unit_content, prev, static_cast<std::uint32_t>('?'), space) ||
        resident_unit_terminal_byte(unit_lengths, unit_content, prev, static_cast<std::uint32_t>('\n'), space)) {
      break;  // hit the prior sentence's end -- this span starts after it
    }
    onset_position = at - 1u;
  }
  const std::uint32_t opener = online_episode_units[onset_position];
  (void)role_canon;
  if (interrogative) {
    atomicAdd(qterm_count + unit, 1u);
    atomicAdd(qonset_count + opener, 1u);
    if (contact_evidence_revision != nullptr &&
        qonset_evidence_revision != nullptr)
      atomicMax(
          reinterpret_cast<unsigned long long*>(
              qonset_evidence_revision + opener),
          static_cast<unsigned long long>(contact_evidence_revision[0]));
  } else {
      decrement_resident_count(qonset_count + opener);
  }
}

// Gather the resident interrogative vocabulary into bounded, walk-reachable
// lists (--qorig). Deterministic ascending-slot-order HOST pass, same idiom
// as collect_subject_field_deterministic just below -- deliberately NOT an
// atomicAdd-into-shared-slot device kernel, because that exact pattern
// (first-come-first-served slot allocation under a capacity bound) is the
// confirmed root cause of a prior nondeterminism bug in this codebase
// (collect_subject_field_kernel, fixed 74c4e987b) and a repeat instance
// caught in early review of an unrelated in-flight prototype this session;
// unit_count is bounded (~10^4-10^5) so a host pass here costs microseconds,
// once per assimilation call, not per generation step.
inline void inject_qorig_units_deterministic(AdultState& state) {
  const std::uint32_t n = state.unit_count;
  cuda_require(cudaMemset(state.qorig_onset_n.get(), 0, state.qorig_onset_n.bytes()),
               "reset qorig onset vocabulary count");
  cuda_require(cudaMemset(state.qorig_term_n.get(), 0, state.qorig_term_n.bytes()),
               "reset qorig terminal vocabulary count");
  if (n == 0u) return;
  std::vector<std::uint32_t> qonset(n), qterm(n), vitality(n);
  cuda_require(cudaMemcpy(qonset.data(), state.qonset_count.get(), n * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "qorig: read qonset_count");
  cuda_require(cudaMemcpy(qterm.data(), state.qterm_count.get(), n * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "qorig: read qterm_count");
  cuda_require(cudaMemcpy(vitality.data(), state.unit_vitality.get(), n * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "qorig: read unit_vitality");
  std::vector<std::uint32_t> onset_ids, onset_w, term_ids, term_w;
  onset_ids.reserve(kQOrigInject); onset_w.reserve(kQOrigInject);
  term_ids.reserve(kQOrigInject); term_w.reserve(kQOrigInject);
  for (std::uint32_t slot = 0u; slot < n; ++slot) {
    const std::uint32_t on = qonset[slot];
    if (onset_ids.size() < kQOrigInject && on >= kQOrigOnsetMin &&
        static_cast<std::uint64_t>(on) * kQOrigSpecific >= vitality[slot]) {
      onset_ids.push_back(slot);
      onset_w.push_back(on);
    }
    const std::uint32_t tm = qterm[slot];
    if (term_ids.size() < kQOrigInject && tm >= kQOrigTermMin) {
      term_ids.push_back(slot);
      term_w.push_back(tm);
    }
  }
  const std::uint32_t onset_count = static_cast<std::uint32_t>(onset_ids.size());
  const std::uint32_t term_count = static_cast<std::uint32_t>(term_ids.size());
  if (onset_count != 0u) {
    cuda_require(cudaMemcpy(state.qorig_onset.get(), onset_ids.data(),
                            onset_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                 "qorig: upload onset ids");
    cuda_require(cudaMemcpy(state.qorig_onset_w.get(), onset_w.data(),
                            onset_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                 "qorig: upload onset weights");
  }
  if (term_count != 0u) {
    cuda_require(cudaMemcpy(state.qorig_term.get(), term_ids.data(),
                            term_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                 "qorig: upload term ids");
    cuda_require(cudaMemcpy(state.qorig_term_w.get(), term_w.data(),
                            term_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                 "qorig: upload term weights");
  }
  cuda_require(cudaMemcpy(state.qorig_onset_n.get(), &onset_count, sizeof(onset_count),
                          cudaMemcpyHostToDevice), "qorig: upload onset vocabulary count");
  cuda_require(cudaMemcpy(state.qorig_term_n.get(), &term_count, sizeof(term_count),
                          cudaMemcpyHostToDevice), "qorig: upload terminal vocabulary count");
}

// Emergent FUNC/NOUN/VERB/OTHER labeling per canonical surface -- the
// surgical sub-diff of 473af804d (its bundled unrelated organs excluded).
// Function words self-identify as the top-frequency canonical surfaces;
// determiners emerge as the FUNC subset whose following-word distribution
// matches the top surface's (cosine over the online bigram stream);
// NOUN/VERB emerge from determiner-adjacency rates. No word list, no
// authored classes -- only resident counts. Host pass, deterministic.
inline void build_unit_pos(AdultState& state) {
  const std::uint32_t n = state.unit_count;
  state.unit_pos.allocate(n);
  if (n == 0u) return;
  std::vector<std::uint32_t> vit(n);
  std::vector<std::uint32_t> ulen(n), ucont(static_cast<std::size_t>(n) * kUnitWords);
  cuda_require(cudaMemcpy(vit.data(), state.unit_vitality.get(),
                          static_cast<std::size_t>(n) * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "pos: read vitality");
  cuda_require(cudaMemcpy(ulen.data(), state.unit_lengths.get(),
                          static_cast<std::size_t>(n) * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "pos: read lengths");
  cuda_require(cudaMemcpy(ucont.data(), state.unit_content.get(),
                          ucont.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "pos: read content");
  auto canon_of = [&](std::uint32_t u) {
    std::string s;
    for (std::uint32_t o = 0u; o < ulen[u]; ++o) {
      const std::uint32_t word = ucont[static_cast<std::size_t>(u) * kUnitWords + o / 4u];
      const std::uint8_t b = static_cast<std::uint8_t>((word >> (8u * (o & 3u))) & 0xffu);
      if (b >= 'A' && b <= 'Z') s.push_back(static_cast<char>(b - 'A' + 'a'));
      else if (b >= 'a' && b <= 'z') s.push_back(static_cast<char>(b));
    }
    return s;
  };
  std::unordered_map<std::string, std::uint32_t> canon_id;
  std::vector<std::uint32_t> unit_canon(n, 0xffffffffu);
  for (std::uint32_t u = 0u; u < n; ++u) {
    const std::string s = canon_of(u);
    if (s.empty()) continue;
    auto it = canon_id.find(s);
    const std::uint32_t cid_value = it != canon_id.end() ? it->second
        : (canon_id.emplace(s, static_cast<std::uint32_t>(canon_id.size())).first->second);
    unit_canon[u] = cid_value;
  }
  const std::uint32_t cn = static_cast<std::uint32_t>(canon_id.size());
  std::vector<std::uint32_t> clen(cn, 0u);
  for (const auto& kv : canon_id) clen[kv.second] = static_cast<std::uint32_t>(kv.first.size());
  std::vector<double> cvit(cn, 0.0);
  for (std::uint32_t u = 0u; u < n; ++u)
    if (unit_canon[u] != 0xffffffffu) cvit[unit_canon[u]] += vit[u];
  const std::uint32_t bc = state.online_bigram_count;
  std::vector<BigramKey> bg(bc);
  std::vector<std::uint32_t> bgc(bc);
  if (bc != 0u) {
    cuda_require(cudaMemcpy(bg.data(), state.online_bigrams.get(),
                            static_cast<std::size_t>(bc) * sizeof(BigramKey),
                            cudaMemcpyDeviceToHost), "pos: read bigrams");
    cuda_require(cudaMemcpy(bgc.data(), state.online_bigram_counts.get(),
                            static_cast<std::size_t>(bc) * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost), "pos: read bigram counts");
  }
  constexpr std::uint32_t kFunc = 50u;
  std::vector<std::uint32_t> order(cn);
  for (std::uint32_t i = 0u; i < cn; ++i) order[i] = i;
  const std::uint32_t keep = std::min(kFunc, cn);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                    [&](std::uint32_t a, std::uint32_t b) { return cvit[a] > cvit[b]; });
  std::vector<char> is_func(cn, 0);
  for (std::uint32_t k = 0u; k < keep; ++k) is_func[order[k]] = 1;
  const auto cid = [&](std::uint32_t u) { return u < n ? unit_canon[u] : 0xffffffffu; };
  std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, double>> fdist;
  for (std::uint32_t i = 0u; i < bc; ++i) {
    const std::uint32_t p = cid(bg[i].previous), q = cid(bg[i].next);
    if (p != 0xffffffffu && q != 0xffffffffu && is_func[p]) fdist[p][q] += bgc[i];
  }
  const std::uint32_t seed = keep != 0u ? order[0] : 0xffffffffu;
  std::vector<double> after_seed(cn, 0.0);
  if (seed != 0xffffffffu)
    for (std::uint32_t i = 0u; i < bc; ++i) {
      const std::uint32_t p = cid(bg[i].previous), q = cid(bg[i].next);
      if (p == seed && q != 0xffffffffu) after_seed[q] += bgc[i];
    }
  std::vector<char> is_det(cn, 0);
  double seed_norm = 0.0;
  std::unordered_map<std::uint32_t, double>* sd = nullptr;
  if (seed != 0xffffffffu && fdist.count(seed)) {
    sd = &fdist[seed]; is_det[seed] = 1;
    for (auto& kv : *sd) seed_norm += kv.second * kv.second;
    seed_norm = std::sqrt(seed_norm);
  }
  for (std::uint32_t k = 0u; sd != nullptr && k < keep; ++k) {
    const std::uint32_t f = order[k];
    if (cvit[f] > 0.0 && after_seed[f] / cvit[f] > 0.03) continue;
    auto it = fdist.find(f); if (it == fdist.end()) continue;
    double dot = 0.0, fn = 0.0;
    for (auto& kv : it->second) { fn += kv.second * kv.second; auto s = sd->find(kv.first); if (s != sd->end()) dot += kv.second * s->second; }
    fn = std::sqrt(fn);
    if (fn > 0.0 && seed_norm > 0.0 && dot / (fn * seed_norm) > 0.12) is_det[f] = 1;
  }
  std::vector<double> after_det(cn, 0.0), before_det(cn, 0.0), in_total(cn, 0.0), out_total(cn, 0.0);
  for (std::uint32_t i = 0u; i < bc; ++i) {
    const std::uint32_t p = cid(bg[i].previous), q = cid(bg[i].next); const double c = bgc[i];
    if (p == 0xffffffffu || q == 0xffffffffu) continue;
    out_total[p] += c; in_total[q] += c;
    if (is_det[p]) after_det[q] += c;
    if (is_det[q]) before_det[p] += c;
  }
  std::vector<std::uint8_t> clabel(cn, 0u);
  for (std::uint32_t u = 0u; u < cn; ++u) {
    if (is_func[u]) { clabel[u] = 1u; continue; }
    if (cvit[u] < 5.0 || clen[u] < 3u) { clabel[u] = 0u; continue; }
    const double ns = after_det[u] / std::max(1.0, in_total[u]);
    const double vs = before_det[u] / std::max(1.0, out_total[u]);
    if (vs > 0.15 && vs > 1.3 * ns) clabel[u] = 3u;        // VERB
    else if (ns > 0.12 && ns > vs) clabel[u] = 2u;         // NOUN
    else clabel[u] = 0u;
  }
  std::vector<std::uint8_t> pos(n, 0u);
  for (std::uint32_t u = 0u; u < n; ++u)
    if (unit_canon[u] != 0xffffffffu) pos[u] = clabel[unit_canon[u]];
  cuda_require(cudaMemcpy(state.unit_pos.get(), pos.data(), n,
                          cudaMemcpyHostToDevice), "pos: upload labels");
}

// Read-only twin of collect_subject_field_deterministic's admission loop. It
// applies the SAME three predicates in the same order and writes nothing but
// the census, so it can run even when the latch itself is gated shut -- which
// is the only way to tell "no unit would have qualified" apart from "the latch
// never ran". Writes no device memory and touches no subject state.
inline void observe_subject_admission(const std::uint32_t* device_cue_scores,
                                      AdultState& state, bool latch_gate_open) {
  // Carry ONLY the fields recorded earlier in this call (entry, episode count,
  // which exit was taken); every scan counter starts from zero, or a second
  // observation in the same call would silently double them.
  SubjectAdmissionCensus census{};
  census.cue_conditioning_entered = state.subject_admission.cue_conditioning_entered;
  census.scoped_episode_count = state.subject_admission.scoped_episode_count;
  census.exact_plan_branch_taken = state.subject_admission.exact_plan_branch_taken;
  census.admitted_subject_count = state.subject_admission.admitted_subject_count;
  census.observed = 1u;
  census.latch_gate_open = latch_gate_open ? 1u : 0u;
  // ⚠ When unit_pos is null the FUNC predicate cannot discriminate and every
  // unit counts as non-FUNC. Recorded so cue_units_nonfunc is never read as
  // evidence that the FUNC test was applied when it was not.
  census.unit_pos_available = state.unit_pos.get() != nullptr ? 1u : 0u;
  const std::uint32_t n = state.unit_count;
  census.cue_units_seen = n;
  if (n == 0u) {
    state.subject_admission = census;
    return;
  }
  std::vector<std::uint32_t> cue_scores(n), unit_lengths(n);
  std::vector<std::uint8_t> unit_pos(n, 0u);
  cuda_require(cudaMemcpy(cue_scores.data(), device_cue_scores,
                          n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "admission census: read cue_scores");
  cuda_require(cudaMemcpy(unit_lengths.data(), state.unit_lengths.get(),
                          n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "admission census: read unit_lengths");
  if (state.unit_pos.get() != nullptr) {
    cuda_require(cudaMemcpy(unit_pos.data(), state.unit_pos.get(), n,
                            cudaMemcpyDeviceToHost),
                 "admission census: read unit_pos");
  }
  for (std::uint32_t unit = 0u; unit < n; ++unit) {
    if (cue_scores[unit] >= kCueNearIdentity) ++census.near_identity_pass_count;
    const bool func =
        state.unit_pos.get() != nullptr && unit_pos[unit] == 1u;
    if (!func) ++census.cue_units_nonfunc;
    if (unit_lengths[unit] < kSubjectMinLen) continue;
    ++census.cue_units_len_ok;
    if (func) continue;
    if (cue_scores[unit] > census.max_nonfunc_len_ok_cue_score)
      census.max_nonfunc_len_ok_cue_score = cue_scores[unit];
  }
  state.subject_admission = census;
}

inline void collect_subject_field_deterministic(
    const std::uint32_t* device_cue_scores, AdultState& state) {
  const std::uint32_t n = state.unit_count;
  cuda_require(cudaMemset(state.subject_count.get(), 0, state.subject_count.bytes()),
               "reset subject_count");
  if (n == 0u) return;
  std::vector<std::uint32_t> cue_scores(n), unit_lengths(n);
  std::vector<std::uint8_t> unit_pos(n, 0u);
  cuda_require(cudaMemcpy(cue_scores.data(), device_cue_scores, n * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "subject: read cue_scores");
  cuda_require(cudaMemcpy(unit_lengths.data(), state.unit_lengths.get(),
                          n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "subject: read unit_lengths");
  if (state.unit_pos.get() != nullptr) {
    cuda_require(cudaMemcpy(unit_pos.data(), state.unit_pos.get(), n, cudaMemcpyDeviceToHost),
                 "subject: read unit_pos");
  }
  std::vector<std::uint32_t> ids, weights;
  ids.reserve(kSubjectCap);
  weights.reserve(kSubjectCap);
  for (std::uint32_t unit = 0u; unit < n && ids.size() < kSubjectCap; ++unit) {
    if (cue_scores[unit] < kCueNearIdentity) continue;
    if (state.unit_pos.get() != nullptr && unit_pos[unit] == 1u) continue;  // FUNC
    const std::uint32_t length = unit_lengths[unit];
    if (length < kSubjectMinLen) continue;
    const std::uint32_t lf = length >= 6u ? 3u : (length >= 4u ? 2u : 1u);
    ids.push_back(unit);
    weights.push_back(kSubjectInitWeight * lf);
  }
  const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
  if (std::getenv("BCC32_FIELD_DIAG") != nullptr) {
    std::vector<std::uint32_t> ucont(static_cast<std::size_t>(n) * kUnitWords);
    cuda_require(cudaMemcpy(ucont.data(), state.unit_content.get(),
                            ucont.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost), "field diag: content");
    std::fprintf(stderr, "subject_field");
    for (std::uint32_t i = 0u; i < count; ++i) {
      const std::uint32_t u = ids[i];
      std::fprintf(stderr, " u%u,w%u,p%u'", u, weights[i], unit_pos[u]);
      for (std::uint32_t o = 0u; o < std::min(unit_lengths[u], 16u); ++o) {
        const std::uint8_t b = static_cast<std::uint8_t>(
            ucont[static_cast<std::size_t>(u) * kUnitWords + o / 4u] >>
            ((o % 4u) * 8u));
        std::fputc(b >= 32u && b < 127u ? b : '.', stderr);
      }
      std::fprintf(stderr, "'");
    }
    std::fprintf(stderr, "\n");
  }
  if (count != 0u) {
    cuda_require(cudaMemcpy(state.subject_ids.get(), ids.data(), count * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice), "subject: upload ids");
    cuda_require(cudaMemcpy(state.subject_weights.get(), weights.data(),
                            count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                 "subject: upload weights");
  }
  cuda_require(cudaMemcpy(state.subject_count.get(), &count, sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "subject: upload count");
}
