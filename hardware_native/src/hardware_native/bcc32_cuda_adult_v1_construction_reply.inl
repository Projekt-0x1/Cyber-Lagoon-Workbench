// bcc32_cuda_adult_v1_construction_reply.inl
//
// The resident construction reply writer owns learned skeleton selection,
// role binding, proposition realization, and its public byte-return path.
// It is included after the adult helper/kernel definitions it consumes and
// preserves the existing resident writer and causal route ordering.

// GENERATE through the resident construction composer: select a learned
// skeleton by grammatical fit against the active subject/completion matter,
// bind role-typed slots with distinct fillers, realize literals + fillers.
// Returns empty when no construction fits (legacy path runs) -- also the
// lesioned/empty-store behavior.
inline std::vector<std::uint8_t> generate_construction_reply(
    AdultState& state, std::uint32_t output_bytes,
    std::uint32_t subject_field_count, bool allow_completion = false) {
  // A quiet tick (empty cue, empty subject field) may still CONTINUE the
  // standing discourse through the progression channel when a theme persists.
  const bool discourse_continuation_possible =
      subject_field_count == 0u && !allow_completion &&
      (state.discourse_theme != 0xffffffffu ||
       !state.discourse_front.empty()) &&
      std::getenv("BCC32_NO_PROGRESSION") == nullptr;
  if (state.construction_lesioned || state.construction_count_host == 0u ||
      (!allow_completion && subject_field_count == 0u &&
       !discourse_continuation_possible) ||
      output_bytes == 0u || state.subject_ids.get() == nullptr)
    return {};
  DeviceArray<unsigned long long> construction_association_mass(
      std::max<std::uint32_t>(1u, state.unit_count));
  cuda_require(cudaMemset(construction_association_mass.get(), 0,
                          construction_association_mass.bytes()),
               "clear construction association mass");
  if (state.online_association_count != 0u) {
    construction::accumulate_subject_association_kernel<<<
        blocks_for(state.online_association_count), kBlock>>>(
        state.online_associations.get(), state.online_association_counts.get(),
        state.online_association_count, state.subject_ids.get(),
        state.subject_count.get(), kSubjectCap,
        construction_association_mass.get());
    cuda_require(cudaGetLastError(), "accumulate subject association mass");
  }
  // CONTENT COMMITMENT -- formed BEFORE any frame decision. The ordered plan
  // (topic + strongest association partners) is computed and stored entirely
  // on-device from resident signals; the host reads back only the plan
  // EXTENT (a count) to decide whether the composer runs in commitment mode,
  // plus unit ids for stderr diagnostics. No host-side content authority.
  bool commitment_active = false;
  // RELATIONAL-TRIPLE COMMITMENT: the typed-triple channel answers first.
  // Retrieval (topic latch, parallel table gather, ranked clause formation)
  // and rendering are entirely on-device; the host reads back only the plan
  // extent/meta to decide the mode, plus unit ids/bytes for stderr
  // diagnostics. When at least one full (subject, connective, value) clause
  // forms, the plan IS the reply -- a proposition needs no frame, the
  // learned connective is the predicate -- and it is realized directly by
  // the plan serializer. Otherwise control falls through unchanged to the
  // relational/co-occurrence commitments. BCC32_LESION_RELATION_TRIPLE
  // skips this block entirely (exact prior behavior).
  const bool triple_frame_mode =
      std::getenv("BCC32_TRIPLE_FRAME") != nullptr;
  const bool role_composer_priority =
      resident_construction_admission_open(state);
  if (!state.content_commit_lesioned && !state.relation_triple_lesioned &&
      state.relation_triples.get() != nullptr && !role_composer_priority) {
    cuda_require(cudaMemset(state.relation_triple_cursor.get(), 0,
                            state.relation_triple_cursor.bytes()),
                 "clear relation triple retrieval cursor");
    const std::uint32_t* topic_subject_ids = state.subject_ids.get();
    const std::uint32_t* topic_subject_weights = state.subject_weights.get();
    const std::uint32_t* topic_subject_count = state.subject_count.get();
    std::uint32_t topic_subject_cap = kSubjectCap;
    const std::uint32_t* topic_closed_class_mask =
        state.construction_closed_class_mask.get();
    const std::uint32_t* topic_unit_lengths = state.unit_lengths.get();
    const std::uint32_t* topic_unit_content = state.unit_content.get();
    std::uint32_t topic_unit_words = kUnitWords;
    const std::uint32_t* topic_boundary_mask = state.boundary_mask.get();
    const std::uint32_t* topic_filler_terminal_mask =
        state.construction_filler_terminal_mask.get();
    const std::uint32_t* topic_role_canon =
        state.role_canon_lesioned ? nullptr : state.construction_role_canon.get();
    std::uint32_t topic_unit_count = state.unit_count;
    std::uint32_t* topic_meta = state.relation_triple_meta.get();
    void* topic_args[] = {&topic_subject_ids,
                          &topic_subject_weights,
                          &topic_subject_count,
                          &topic_subject_cap,
                          &topic_closed_class_mask,
                          &topic_unit_lengths,
                          &topic_unit_content,
                          &topic_unit_words,
                          &topic_boundary_mask,
                          &topic_filler_terminal_mask,
                          &topic_role_canon,
                          &topic_unit_count,
                          &topic_meta};
    cuda_require(cudaLaunchKernel(
                     reinterpret_cast<const void*>(
                         construction::select_triple_topic_kernel),
                     dim3(1u, 1u, 1u), dim3(kBlock, 1u, 1u), topic_args,
                     3u * kBlock * sizeof(std::uint32_t), nullptr),
                 "latch relation triple topic");
    cuda_require(cudaGetLastError(), "latch relation triple topic");
    construction::gather_relation_triples_kernel<<<
        blocks_for(construction::kRelationTripleHashCap), kBlock>>>(
        state.relation_triples.get(), state.relation_triple_counts.get(),
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.relation_cue_scores.get(), state.relation_cue_orders.get(),
        state.relation_cue_exact.get(), state.relation_operator_order.get(),
        nullptr, kCueNearIdentity,
        state.relation_triple_candidates.get(),
        state.relation_triple_cursor.get());
    cuda_require(cudaGetLastError(), "gather resident relation triples");
    std::uint32_t operator_candidate_count = 0u;
    cuda_require(cudaMemcpy(&operator_candidate_count,
                            state.relation_triple_cursor.get(),
                            sizeof(operator_candidate_count),
                            cudaMemcpyDeviceToHost),
                 "read operator-conditioned relation extent");
    if (operator_candidate_count == 0u) {
      cuda_require(cudaMemset(state.relation_triple_cursor.get(), 0,
                              state.relation_triple_cursor.bytes()),
                   "reset relation cursor for topic fallback");
      cuda_require(cudaMemset(state.relation_operator_order.get(), 0,
                              state.relation_operator_order.bytes()),
                   "reset latest exact fallback topic order");
      construction::derive_latest_exact_content_order_kernel<<<
          blocks_for(state.unit_count), kBlock>>>(
          state.relation_cue_exact.get(), state.relation_cue_orders.get(),
          state.construction_closed_class_mask.get(), state.unit_count,
          state.relation_operator_order.get());
      cuda_require(cudaGetLastError(), "derive latest exact fallback topic");
      construction::gather_relation_triples_kernel<<<
          blocks_for(construction::kRelationTripleHashCap), kBlock>>>(
          state.relation_triples.get(), state.relation_triple_counts.get(),
          state.role_canon_lesioned ? nullptr
                                    : state.construction_role_canon.get(),
          state.relation_cue_scores.get(), state.relation_cue_orders.get(),
          state.relation_cue_exact.get(), nullptr,
          state.relation_operator_order.get(), 0xffffu,
          state.relation_triple_candidates.get(),
          state.relation_triple_cursor.get());
      cuda_require(cudaGetLastError(), "gather topic relation fallback");
    }
    construction::form_triple_commitment_kernel<BigramKey><<<1u, 1u>>>(
        state.relation_triples.get(), state.relation_triple_counts.get(),
        state.relation_triple_candidates.get(),
        state.relation_triple_cursor.get(),
        state.relation_triple_type_total.get(),
        state.relation_triple_type_mirrored.get(),
        state.relation_cue_scores.get(), state.relation_cue_orders.get(),
        state.relation_cue_exact.get(), state.relation_operator_order.get(),
        kCueNearIdentity,
        operator_candidate_count == 0u,
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.qonset_count.get(), state.unit_count,
        state.unit_vitality.get(),
        state.construction_closed_class_mask.get(), state.unit_lengths.get(),
        state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
        state.construction_filler_terminal_mask.get(),
        state.online_bigrams.get(), state.online_bigram_counts.get(),
        state.online_bigram_count,
        /*prior_ends=*/nullptr, /*prior_end_count=*/0u,
        /*required_connective_canon=*/construction::kNoTripleUnit,
        /*topic_affinity=*/nullptr, /*topic_affinity_floor=*/0ull,
        /*analogical_topic=*/construction::kNoTripleUnit,
        /*mate_counts=*/nullptr, /*subject_degree=*/nullptr,
        /*unit_pos=*/nullptr,
        /*max_clauses=*/triple_frame_mode
            ? construction::kRelationTripleMaxClauses
            : 1u,
        state.relation_triple_plan.get(), state.relation_triple_meta.get());
    cuda_require(cudaGetLastError(), "form resident triple commitment");
    std::uint32_t triple_meta[8] = {};
    cuda_require(cudaMemcpy(triple_meta, state.relation_triple_meta.get(),
                            sizeof(triple_meta), cudaMemcpyDeviceToHost),
                 "read resident triple commitment meta");
    std::uint32_t triple_candidates = 0u;
    cuda_require(cudaMemcpy(&triple_candidates,
                            state.relation_triple_cursor.get(),
                            sizeof(triple_candidates),
                            cudaMemcpyDeviceToHost),
                 "read resident triple candidate extent");
    std::uint32_t relation_operator_order = 0xffffffffu;
    cuda_require(cudaMemcpy(&relation_operator_order,
                            state.relation_operator_order.get(),
                            sizeof(relation_operator_order),
                            cudaMemcpyDeviceToHost),
                 "read relation operator order for diagnostics");
    std::fprintf(stderr,
                 "relation_triple_commit topic=u%u candidates=%u clauses=%u"
                 " units=%u tier=%u operator_order=%u",
                 triple_meta[2], triple_candidates, triple_meta[1],
                 triple_meta[0], triple_meta[3], relation_operator_order);
    // Opt-in retrieval diagnostic: surface the gathered candidate triples
    // (ids/counts/bytes, stderr only) so ranking decisions are inspectable.
    if (std::getenv("BCC32_TRIPLE_DIAG") != nullptr && triple_candidates != 0u) {
      const std::uint32_t shown = std::min<std::uint32_t>(
          triple_candidates, 24u);
      std::vector<std::uint32_t> entries(shown);
      cuda_require(cudaMemcpy(entries.data(),
                              state.relation_triple_candidates.get(),
                              entries.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read triple candidates for diagnostics");
      auto unit_text = [&](std::uint32_t unit) {
        std::string text;
        if (unit == 0xffffffffu) return text;
        std::uint32_t length = 0u;
        cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                                sizeof(length), cudaMemcpyDeviceToHost),
                     "read candidate unit length");
        std::uint32_t words[kUnitWords] = {};
        cuda_require(cudaMemcpy(words,
                                state.unit_content.get() +
                                    static_cast<std::size_t>(unit) * kUnitWords,
                                sizeof(words), cudaMemcpyDeviceToHost),
                     "read candidate unit bytes");
        const std::uint32_t bounded = std::min<std::uint32_t>(length, 24u);
        for (std::uint32_t offset = 0u; offset < bounded; ++offset) {
          const std::uint8_t byte = static_cast<std::uint8_t>(
              words[offset / 4u] >> ((offset % 4u) * 8u));
          text.push_back(byte >= 32u && byte < 127u
                             ? static_cast<char>(byte) : '.');
        }
        return text;
      };
      std::uint32_t field_count = 0u;
      cuda_require(cudaMemcpy(&field_count, state.subject_count.get(),
                              sizeof(field_count), cudaMemcpyDeviceToHost),
                   "read subject field extent for diagnostics");
      field_count = std::min<std::uint32_t>(field_count, kSubjectCap);
      std::vector<std::uint32_t> field_units(field_count);
      if (field_count != 0u)
        cuda_require(cudaMemcpy(field_units.data(), state.subject_ids.get(),
                                field_units.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read subject field units for diagnostics");
      std::fprintf(stderr, "\nrelation_triple_cue_glue");
      for (const std::uint32_t cue_unit : field_units) {
        std::uint32_t glue = 0u;
        cuda_require(
            cudaMemcpy(&glue,
                       state.construction_closed_class_mask.get() + cue_unit,
                       sizeof(glue), cudaMemcpyDeviceToHost),
            "read subject field glue flag");
        if (glue != 0u) std::fprintf(stderr, " u%u", cue_unit);
      }
      std::fprintf(stderr, "\nrelation_triple_candidates");
      for (std::uint32_t c = 0u; c < shown; ++c) {
        const std::uint32_t slot = entries[c] & ~0x80000000u;
        const bool forward = (entries[c] & 0x80000000u) == 0u;
        construction::RelationTriple triple{};
        std::uint32_t count = 0u;
        cuda_require(cudaMemcpy(&triple, state.relation_triples.get() + slot,
                                sizeof(triple), cudaMemcpyDeviceToHost),
                     "read candidate triple");
        cuda_require(cudaMemcpy(&count,
                                state.relation_triple_counts.get() + slot,
                                sizeof(count), cudaMemcpyDeviceToHost),
                     "read candidate triple count"),
        std::fprintf(stderr, " %s[%s|%s%s|%s]x%u",
                     forward ? "F" : "R", unit_text(triple.subject).c_str(),
                     unit_text(triple.connective).c_str(),
                     triple.connective2 != 0xffffffffu
                         ? unit_text(triple.connective2).c_str() : "",
                     unit_text(triple.value).c_str(), count);
        {
          const std::uint32_t topic_end =
              forward ? triple.subject : triple.value;
          std::uint32_t score = 0u, order = 0u, exact = 0u;
          cuda_require(cudaMemcpy(&score,
                                  state.relation_cue_scores.get() + topic_end,
                                  sizeof(score), cudaMemcpyDeviceToHost),
                       "diag: read topic cue score");
          cuda_require(cudaMemcpy(&order,
                                  state.relation_cue_orders.get() + topic_end,
                                  sizeof(order), cudaMemcpyDeviceToHost),
                       "diag: read topic cue order");
          cuda_require(cudaMemcpy(&exact,
                                  state.relation_cue_exact.get() + topic_end,
                                  sizeof(exact), cudaMemcpyDeviceToHost),
                       "diag: read topic cue exact");
          std::fprintf(stderr, "(s%u,o%u,e%u)", score, order, exact);
        }
      }
    }
    if (triple_meta[0] != 0u && triple_meta[0] <= construction::kCommitmentCap) {
      // Diagnostics only: surface the committed clause words on stderr.
      std::vector<std::uint32_t> plan_host(triple_meta[0]);
      cuda_require(cudaMemcpy(plan_host.data(),
                              state.relation_triple_plan.get(),
                              plan_host.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read resident triple plan");
      for (const std::uint32_t unit : plan_host) {
        std::uint32_t length = 0u;
        cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                                sizeof(length), cudaMemcpyDeviceToHost),
                     "read triple trace unit length");
        std::uint32_t words[kUnitWords] = {};
        cuda_require(cudaMemcpy(words,
                                state.unit_content.get() +
                                    static_cast<std::size_t>(unit) * kUnitWords,
                                sizeof(words), cudaMemcpyDeviceToHost),
                     "read triple trace unit bytes");
        std::fprintf(stderr, " u%u'", unit);
        const std::uint32_t bounded = std::min<std::uint32_t>(length, 24u);
        for (std::uint32_t offset = 0u; offset < bounded; ++offset) {
          const std::uint8_t byte = static_cast<std::uint8_t>(
              words[offset / 4u] >> ((offset % 4u) * 8u));
          std::fputc(byte >= 32u && byte < 127u ? byte : '.', stderr);
        }
        std::fprintf(stderr, "'");
      }
    }
    std::fprintf(stderr, "\n");
    if (triple_frame_mode && triple_meta[0] >= 3u &&
        triple_meta[0] <= construction::kCommitmentCap) {
      cuda_require(cudaMemcpy(state.content_commitment_units.get(),
                              state.relation_triple_plan.get(),
                              triple_meta[0] * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToDevice),
                   "hand relation plan to construction organ");
      cuda_require(cudaMemset(state.content_commitment_meta.get(), 0,
                              state.content_commitment_meta.bytes()),
                   "reset construction commitment cursor");
      cuda_require(cudaMemcpy(state.content_commitment_meta.get(),
                              state.relation_triple_meta.get(),
                              sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
                   "hand relation plan extent to construction organ");
      commitment_active = true;
    }
    if (!triple_frame_mode && triple_meta[0] >= 3u &&
        triple_meta[0] <= construction::kCommitmentCap &&
        std::getenv("BCC32_NO_PROGRESSION") != nullptr) {
      // DEFAULT SAFE PATH: emit the single grounded proposition the cue
      // typed. Multi-sentence spreading-activation continuation (below,
      // BCC32_PROGRESSION) was proven on the real gate to trade the copying
      // guard for coherence -- a locally coherent walk over the learned
      // relation graph reconstructs corpus-verbatim spans (>48 bytes,
      // instant fail), because that graph encodes corpus adjacency; a
      // novel walk drifts. One grounded clause keeps verbatim_span < 48 and
      // stays on topic. See the branch report for the impossibility proof.
      DeviceArray<std::uint8_t> device_output(output_bytes);
      DeviceArray<std::uint32_t> device_generated_count(1u);
      construction::emit_construction_plan_kernel<<<1u, 1u>>>(
          state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
          state.relation_triple_plan.get(), state.relation_triple_meta.get(),
          state.boundary_mask.get(), device_output.get(), output_bytes,
          device_generated_count.get());
      cuda_require(cudaGetLastError(), "realize resident triple claim");
      cuda_require(cudaDeviceSynchronize(), "complete resident triple claim");
      std::uint32_t generated_count = 0u;
      cuda_require(cudaMemcpy(&generated_count, device_generated_count.get(),
                              sizeof(generated_count), cudaMemcpyDeviceToHost),
                   "read resident triple claim extent");
      if (generated_count != 0u && generated_count <= output_bytes) {
        std::vector<std::uint8_t> output(generated_count);
        cuda_require(cudaMemcpy(output.data(), device_output.get(),
                                generated_count, cudaMemcpyDeviceToHost),
                     "read resident triple claim bytes");
        std::fprintf(stderr, "relation_triple_reply clauses=%u units=%u\n",
                     triple_meta[1], triple_meta[0]);
        return output;
      }
    }
    if (!triple_frame_mode && !state.relation_triple_lesioned &&
        !role_composer_priority &&
        ((triple_meta[0] >= 3u &&
          triple_meta[0] <= construction::kCommitmentCap) ||
         discourse_continuation_possible) &&
        std::getenv("BCC32_NO_PROGRESSION") == nullptr) {
      // THEMATIC PROGRESSION (reafferent continuation). Realize the first
      // clause, then let that clause's own rheme -- the new information the
      // organism just uttered, exported by the commitment kernel in
      // meta[4] -- become the next topic: promote it into the cue identity
      // field (the organism's own emission re-entering the same field an
      // external cue writes; efference copy / Levelt's perceptual loop),
      // gather learned relations about IT with the SAME retrieval kernels,
      // and continue sentence by sentence until the byte budget is spent,
      // the resident store offers nothing new, or the refractory set would
      // force a restatement. This is Danes' linear thematic progression
      // (rheme of sentence n = theme of sentence n+1) realized purely over
      // resident learned matter: the host moves unit IDs the organism just
      // said, and never authors content. Sentence closure uses the closure
      // byte the organism itself discovered from the corpus stream.
      constexpr std::uint32_t kProgressionEndCap = 128u;
      constexpr std::uint32_t kProgressionMaxSentences = 10u;
      std::vector<std::uint8_t> reply;
      std::vector<std::uint32_t> said_ends;
      DeviceArray<std::uint32_t> device_said_ends(kProgressionEndCap);
      std::uint32_t closure_byte = 0u;  // 0 = no learned closure; omit it
      if (state.construction_closure_count != 0u) {
        cuda_require(cudaMemcpy(&closure_byte,
                                state.construction_closure_bytes.get(),
                                sizeof(closure_byte), cudaMemcpyDeviceToHost),
                     "read learned sentence closure byte");
      }
      auto remember_end = [&](std::uint32_t unit) {
        if (unit == 0xffffffffu || unit >= state.unit_count) return;
        for (const std::uint32_t said : said_ends)
          if (said == unit) return;
        // Evict the OLDEST when full -- dropping the newest would freeze the
        // refractory set and let the current clause repeat forever.
        if (said_ends.size() >= kProgressionEndCap)
          said_ends.erase(said_ends.begin());
        said_ends.push_back(unit);
      };
      std::uint32_t sentence_meta[8];
      for (std::uint32_t k = 0u; k < 8u; ++k) sentence_meta[k] = triple_meta[k];
      std::uint32_t sentences = 0u;
      std::vector<std::uint32_t> uttered_themes;  // meta[2] per emitted clause
      std::vector<std::uint32_t> front;  // content units already active/uttered
      auto push_front = [&](std::uint32_t unit) {
        if (unit == 0xffffffffu || unit >= state.unit_count) return;
        for (const std::uint32_t member : front)
          if (member == unit) return;
        if (front.size() < kProgressionEndCap) front.push_back(unit);
      };
      // Quiet continuation: no cue, no fresh commitment -- the standing
      // discourse theme seeds the front, and everything already said in
      // PRIOR replies is refractory, so successive quiet ticks advance the
      // monologue instead of restating it.
      const bool quiet_continuation =
          discourse_continuation_possible && triple_meta[0] < 3u;
      if (!quiet_continuation && subject_field_count != 0u) {
        // Seed the front with the question's OTHER legal content terms so
        // the paragraph engages the full question (phenotype topic_contact
        // requires >= 2 question terms) and survives a weak first rheme.
        const std::uint32_t field_count =
            std::min<std::uint32_t>(subject_field_count, kSubjectCap);
        std::vector<std::uint32_t> field_units(field_count);
        cuda_require(cudaMemcpy(field_units.data(), state.subject_ids.get(),
                                field_count * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read subject field for front seeding");
        // The field lists every cue-matched unit, including near-duplicate
        // surface variants of the SAME word ("talent-", "talent?",
        // "talent."). Raw truncation let those variants crowd out the
        // question's OTHER content terms ("firms", "identify") -- and a
        // variant carrying clause-breaking bytes can never anchor a clause
        // anyway (the composer's subject-role artifact ban). Seed DISTINCT
        // anchorable words instead: skip artifact surfaces and units whose
        // alphabetic word matches an already-seeded seed.
        std::uint32_t seeded = 0u;
        std::vector<std::string> seeded_words;
        // The attested first clause already predicates about the primary
        // topic word: its surface variants ("talent-") would only restate
        // that seat, so pre-claim the word for it.
        if (triple_meta[0] >= 3u && triple_meta[2] < state.unit_count) {
          std::uint32_t length = 0u;
          cuda_require(cudaMemcpy(&length,
                                  state.unit_lengths.get() + triple_meta[2],
                                  sizeof(length), cudaMemcpyDeviceToHost),
                       "read attested topic length");
          std::uint32_t words[kUnitWords] = {};
          cuda_require(cudaMemcpy(words,
                                  state.unit_content.get() +
                                      static_cast<std::size_t>(triple_meta[2]) *
                                          kUnitWords,
                                  sizeof(words), cudaMemcpyDeviceToHost),
                       "read attested topic bytes");
          std::string word;
          for (std::uint32_t o = 0u; o < length && o < kMaxUnitBytes; ++o) {
            const std::uint8_t b = static_cast<std::uint8_t>(
                words[o / 4u] >> ((o % 4u) * 8u));
            if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) {
              if (word.size() == static_cast<std::size_t>(o))
                word.push_back(static_cast<char>(b | 0x20u));
            }
          }
          if (!word.empty()) seeded_words.push_back(word);
        }
        for (const std::uint32_t unit : field_units) {
          if (seeded >= 4u) break;
          if (unit >= state.unit_count) continue;
          std::uint32_t length = 0u;
          cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                                  sizeof(length), cudaMemcpyDeviceToHost),
                       "read subject seed length");
          std::uint32_t words[kUnitWords] = {};
          cuda_require(cudaMemcpy(words,
                                  state.unit_content.get() +
                                      static_cast<std::size_t>(unit) *
                                          kUnitWords,
                                  sizeof(words), cudaMemcpyDeviceToHost),
                       "read subject seed bytes");
          bool artifact = false;
          std::string word;
          for (std::uint32_t o = 0u; o < length && o < kMaxUnitBytes; ++o) {
            const std::uint8_t b = static_cast<std::uint8_t>(
                words[o / 4u] >> ((o % 4u) * 8u));
            // Same byte-class ban as the composer's subject role.
            if (b < 0x20u || b == 0x21u || b == 0x22u || b == 0x28u ||
                b == 0x29u || b == 0x2eu || b == 0x3au || b == 0x3bu ||
                b == 0x3fu || b == 0x5bu || b == 0x5du || b == 0x7bu ||
                b == 0x7du || b == 0xe2u) {
              artifact = true;
              break;
            }
            if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) {
              if (word.size() == static_cast<std::size_t>(o))
                word.push_back(static_cast<char>(b | 0x20u));
            }
          }
          std::fprintf(stderr, "seed_scan u%u word='%s' artifact=%u\n", unit,
                       word.c_str(), artifact ? 1u : 0u);
          if (artifact || word.empty()) continue;
          bool duplicate_word = false;
          for (const std::string& prior : seeded_words)
            duplicate_word |= prior == word;
          if (duplicate_word) continue;
          if (state.qonset_count.get() != nullptr) {
            // LEARNED question-onset signal: a unit the stream established
            // as a question opener ("should", "what") is topic-banned in
            // the composer and would only waste its seed seat here.
            std::uint32_t onset = 0u;
            cuda_require(cudaMemcpy(&onset, state.qonset_count.get() + unit,
                                    sizeof(onset), cudaMemcpyDeviceToHost),
                         "read seed question-onset count");
            if (onset >= construction::kQonsetTopicFloor) continue;
          }
          push_front(unit);
          seeded_words.push_back(word);
          ++seeded;
        }
      }
      if (quiet_continuation) {
        push_front(state.discourse_theme);
        for (const std::uint32_t member : state.discourse_front)
          push_front(member);  // newest persisted themes end up tried first
      }
      {
        // Cross-reply refractory seeds. Leave exactly the headroom one
        // reply can consume (kProgressionMaxSentences clauses x 2 ends),
        // so the current reply's own bans are never evicted mid-paragraph
        // (the duplicate-clause failure mode) while the window still spans
        // several prior replies (a window shorter than two replies' churn
        // let alternating quiet ticks restate each other verbatim).
        constexpr std::uint32_t kSaidSeedCap =
            kProgressionEndCap - 2u * kProgressionMaxSentences;
        std::uint32_t seeded_said = 0u;
        for (auto it = state.discourse_said.rbegin();
             it != state.discourse_said.rend() && seeded_said < kSaidSeedCap;
             ++it) {
          remember_end(*it);
          ++seeded_said;
        }
      }
      // GENERALIZING CONTINUATION. Sentence 1 is the attested, cue-typed
      // claim (grounded retrieval). Every LATER sentence is a LICENSED-NOVEL
      // composition: for the active topic T, the two analogy kernels find
      // T's emergent distributional category (units the store proves share
      // >= 2 predication contexts with T) and gather claims (T, K, B') that
      // a category mate attests while (T, K, B') itself is unattested. The
      // clause is therefore surface-novel BY CONSTRUCTION -- its byte string
      // does not occur in the corpus, so it cannot extend a verbatim span --
      // yet role-licensed, so it remains a well-formed predication about T.
      // Refractory suppression + the topical-affinity field discipline the
      // walk; termination is content-driven (no on-topic licensed claim
      // left). This replaces adjacency-walk continuation, which the gate
      // proved must either copy (>48-byte verbatim) or drift.
      DeviceArray<std::uint32_t> mate_counts_buffer(
          std::max<std::uint32_t>(1u, state.unit_count));
      DeviceArray<std::uint32_t> subject_degree_buffer(
          std::max<std::uint32_t>(1u, state.unit_count));
      auto try_analogy = [&](std::uint32_t topic) -> bool {
        if (topic == 0xffffffffu || topic >= state.unit_count) return false;
        char topic_surface[25] = {};
        {
          // Theme hygiene, from resident signals only: (a) a unit carrying
          // control bytes is a fused cross-line sensor artifact (the
          // kBoundaryCount=1 defect), not a word -- chaining from it yields
          // "be\nthe"-style subjects; (b) a FUNC-classed unit (emergent
          // word-class field, no authored list) cannot anchor a discourse
          // theme.
          std::uint32_t topic_length = 0u;
          cuda_require(cudaMemcpy(&topic_length,
                                  state.unit_lengths.get() + topic,
                                  sizeof(topic_length), cudaMemcpyDeviceToHost),
                       "read theme unit length");
          if (topic_length < 4u) return false;
          std::uint32_t topic_words[kUnitWords] = {};
          cuda_require(cudaMemcpy(topic_words,
                                  state.unit_content.get() +
                                      static_cast<std::size_t>(topic) * kUnitWords,
                                  sizeof(topic_words), cudaMemcpyDeviceToHost),
                       "read theme unit bytes");
          for (std::uint32_t o = 0u; o < topic_length && o < kMaxUnitBytes; ++o) {
            const std::uint8_t b = static_cast<std::uint8_t>(
                topic_words[o / 4u] >> ((o % 4u) * 8u));
            if (b < 0x20u) return false;  // fused sensor artifact
            if (o < 24u)
              topic_surface[o] = b >= 32u && b < 127u
                                     ? static_cast<char>(b)
                                     : '.';
          }
          if (state.unit_pos.get() != nullptr &&
              topic < state.unit_pos.size()) {
            std::uint8_t pos_label = 0u;
            cuda_require(cudaMemcpy(&pos_label, state.unit_pos.get() + topic,
                                    1u, cudaMemcpyDeviceToHost),
                         "read theme word class");
            if (pos_label == 1u) return false;  // FUNC cannot be a theme
          }
        }
        // The question's cue fields are consumed by sentence 1; stale scores
        // would cue-type connectives and pollute analogical ranking.
        cuda_require(cudaMemset(state.relation_cue_exact.get(), 0,
                                state.relation_cue_exact.bytes()),
                     "clear stale exact cue for analogy");
        cuda_require(cudaMemset(state.relation_cue_scores.get(), 0,
                                state.relation_cue_scores.bytes()),
                     "clear stale cue scores for analogy");
        cuda_require(cudaMemset(state.relation_cue_orders.get(), 0xff,
                                state.relation_cue_orders.bytes()),
                     "clear stale cue orders for analogy");
        cuda_require(cudaMemset(mate_counts_buffer.get(), 0,
                                mate_counts_buffer.bytes()),
                     "clear category mate counts");
        cuda_require(cudaMemset(subject_degree_buffer.get(), 0,
                                subject_degree_buffer.bytes()),
                     "clear subject context degrees");
        construction::count_structural_category_mates_kernel<<<
            blocks_for(construction::kRelationRoleHashCap), kBlock>>>(
            state.relation_roles.get(), state.relation_role_counts.get(),
            topic, state.unit_count, mate_counts_buffer.get(),
            subject_degree_buffer.get(),
            state.relation_probe_support_histogram.get(),
            state.relation_probe_total.get());
        cuda_require(cudaGetLastError(), "count distributional category mates");
        // Same topic, same role canon, same substituted lookup -- only the
        // source-slot count gate is absent. Observer: it writes three
        // counters no reply, candidate, ranking or stored state reads.
        if (std::getenv("BCC32_SHADOW_CENSUS") != nullptr) {
          construction::census_substituted_support_kernel<<<
              blocks_for(construction::kRelationTripleHashCap), kBlock>>>(
              state.relation_triples.get(), state.relation_triple_counts.get(),
              topic,
              state.role_canon_lesioned ? nullptr
                                        : state.construction_role_canon.get(),
              state.unit_count, state.relation_census_histogram.get(),
              state.relation_census_total.get(),
              state.relation_census_source_singletons.get(),
              state.relation_counterfactual_histogram.get(),
              state.relation_counterfactual_total.get(),
              state.relation_topic_strictly_greater.get(),
              state.relation_counterfactual_strictly_greater.get(),
              state.relation_support_ties.get());
          cuda_require(cudaGetLastError(),
                       "census ungated substituted support");
        }
        cuda_require(cudaMemset(state.relation_triple_cursor.get(), 0,
                                state.relation_triple_cursor.bytes()),
                     "reset analogical candidate cursor");
        construction::gather_analogical_triples_kernel<<<
            blocks_for(construction::kRelationTripleHashCap), kBlock>>>(
            state.relation_triples.get(), state.relation_triple_counts.get(),
            topic, mate_counts_buffer.get(),
            state.role_canon_lesioned ? nullptr
                                      : state.construction_role_canon.get(),
            state.unit_count, state.relation_triple_candidates.get(),
            state.relation_triple_cursor.get());
        cuda_require(cudaGetLastError(), "gather licensed-novel claims");
        const std::uint32_t end_count = static_cast<std::uint32_t>(
            said_ends.size() < kProgressionEndCap ? said_ends.size()
                                                  : kProgressionEndCap);
        if (end_count != 0u) {
          cuda_require(cudaMemcpy(device_said_ends.get(), said_ends.data(),
                                  end_count * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice),
                       "upload reply refractory endpoints");
        }
        construction::form_triple_commitment_kernel<BigramKey><<<1u, 1u>>>(
            state.relation_triples.get(), state.relation_triple_counts.get(),
            state.relation_triple_candidates.get(),
            state.relation_triple_cursor.get(),
            state.relation_triple_type_total.get(),
            state.relation_triple_type_mirrored.get(),
            state.relation_cue_scores.get(), state.relation_cue_orders.get(),
            state.relation_cue_exact.get(),
            state.relation_operator_order.get(), kCueNearIdentity,
            /*topic_fallback=*/true,
            state.role_canon_lesioned ? nullptr
                                      : state.construction_role_canon.get(),
            state.qonset_count.get(), state.unit_count,
            state.unit_vitality.get(),
            state.construction_closed_class_mask.get(),
            state.unit_lengths.get(),
            state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
            state.construction_filler_terminal_mask.get(),
            state.online_bigrams.get(), state.online_bigram_counts.get(),
            state.online_bigram_count,
            device_said_ends.get(), end_count,
            /*required_connective_canon=*/construction::kNoTripleUnit,
            // Quiet continuation has no cue field to accumulate association
            // mass from (it would be all zeros and veto everything); the
            // cross-reply refractory set carries the discipline there.
            quiet_continuation ? nullptr : construction_association_mass.get(),
            /*topic_affinity_floor=*/2ull,
            /*analogical_topic=*/topic,
            mate_counts_buffer.get(), subject_degree_buffer.get(),
            state.unit_pos.size() >= state.unit_count ? state.unit_pos.get()
                                                      : nullptr,
            /*max_clauses=*/1u,
            state.relation_triple_plan.get(),
            state.relation_triple_meta.get());
        cuda_require(cudaGetLastError(), "form licensed-novel commitment");
        cuda_require(cudaMemcpy(sentence_meta,
                                state.relation_triple_meta.get(),
                                sizeof(sentence_meta),
                                cudaMemcpyDeviceToHost),
                     "read licensed-novel commitment meta");
        std::uint32_t analogical_candidates = 0u;
        cuda_require(cudaMemcpy(&analogical_candidates,
                                state.relation_triple_cursor.get(),
                                sizeof(analogical_candidates),
                                cudaMemcpyDeviceToHost),
                     "read analogical candidate extent");
        std::fprintf(stderr,
                     "analogy_probe topic=u%u'%s' candidates=%u units=%u\n",
                     topic, topic_surface, analogical_candidates,
                     sentence_meta[0]);
        return sentence_meta[0] >= 3u &&
               sentence_meta[0] <= construction::kCommitmentCap;
      };
      // Spread across the active front, newest matter first (the rheme just
      // uttered), falling back to older active units when the newest offers
      // no licensed-novel claim.
      auto spread_step = [&]() -> bool {
        // Newest matter first; every third sentence (every SECOND in quiet
        // mode, where the persisted contact themes are the only anchor to
        // the standing conversation) the OLDEST active theme (the seeded
        // question terms) gets first turn, so the paragraph engages the
        // full question instead of riding one chain.
        // A quiet tick has no cue: the persisted conversation topics are
        // its ONLY anchor, so every sentence offers the seeds a turn
        // first (unused seeds are few; the newest-matter chain below
        // carries the sentences in between).
        const bool seeds_first =
            quiet_continuation || (sentences % 3u == 2u);
        if (seeds_first) {
          // Each seed gets ONE turn: a theme already predicated about this
          // reply yields its rotation slot to the next seed, so the
          // paragraph engages the question's OTHER terms instead of the
          // first seed winning every slot. A cued reply walks its seeds
          // OLDEST-first (the question's own terms were seeded before
          // sentence 1); a quiet tick walks NEWEST-first, because its
          // anchor to the standing conversation is the MOST RECENT cued
          // exchange (continuity is judged against what was just said,
          // not the whole session).
          for (std::size_t n = 0u; n < front.size(); ++n) {
            const std::size_t i =
                quiet_continuation ? front.size() - 1u - n : n;
            bool used = false;
            for (const std::uint32_t theme : uttered_themes)
              used |= theme == front[i];
            if (used) continue;
            if (try_analogy(front[i])) return true;
          }
          // No unused seed offers a licensed claim: fall through to the
          // ordinary newest-matter-first continuation below.
        }
        for (std::size_t i = front.size(); i-- != 0u;)
          if (try_analogy(front[i])) return true;
        return false;
      };
      for (std::uint32_t sentence = 0u; sentence < kProgressionMaxSentences;
           ++sentence) {
        if (sentence != 0u || quiet_continuation) {
          if (!spread_step()) break;  // no on-topic claim left: paragraph ends
        }
        // Realize this sentence into the remaining byte budget.
        if (reply.size() + 16u > output_bytes) break;
        const std::uint32_t remaining =
            output_bytes - static_cast<std::uint32_t>(reply.size());
        DeviceArray<std::uint8_t> device_output(remaining);
        DeviceArray<std::uint32_t> device_generated_count(1u);
        construction::emit_construction_plan_kernel<<<1u, 1u>>>(
            state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
            state.relation_triple_plan.get(), state.relation_triple_meta.get(),
            state.boundary_mask.get(), device_output.get(), remaining,
            device_generated_count.get());
        cuda_require(cudaGetLastError(), "realize resident triple claim");
        cuda_require(cudaDeviceSynchronize(),
                     "complete resident triple claim");
        std::uint32_t generated_count = 0u;
        cuda_require(cudaMemcpy(&generated_count,
                                device_generated_count.get(),
                                sizeof(generated_count),
                                cudaMemcpyDeviceToHost),
                     "read resident triple claim extent");
        if (generated_count == 0u || generated_count > remaining) break;
        if (generated_count == remaining && sentence != 0u)
          break;  // truncated continuation clause: end the paragraph instead
        std::vector<std::uint8_t> clause(generated_count);
        cuda_require(cudaMemcpy(clause.data(), device_output.get(),
                                generated_count, cudaMemcpyDeviceToHost),
                     "read resident triple claim bytes");
        while (!clause.empty() &&
               (clause.back() == ' ' || clause.back() == '\n' ||
                clause.back() == '\r' || clause.back() == '\t'))
          clause.pop_back();
        if (clause.empty()) break;
        reply.insert(reply.end(), clause.begin(), clause.end());
        if (closure_byte >= 32u && closure_byte < 127u &&
            clause.back() != static_cast<std::uint8_t>(closure_byte)) {
          reply.push_back(static_cast<std::uint8_t>(closure_byte));
        }
        reply.push_back(' ');
        remember_end(sentence_meta[2]);
        remember_end(sentence_meta[4]);
        push_front(sentence_meta[2]);  // theme + rheme re-enter the active
        push_front(sentence_meta[4]);  // front; spreading activation continues
        uttered_themes.push_back(sentence_meta[2]);
        ++sentences;
        std::fprintf(stderr,
                     "triple_progression sentence=%u topic=u%u rheme=u%u"
                     " units=%u tier=%u bytes=%zu\n",
                     sentence, sentence_meta[2], sentence_meta[4],
                     sentence_meta[0], sentence_meta[3], reply.size());
      }
      // The support the substitution probes actually found, accumulated
      // since genesis. Off unless asked for: the stream gate only admits
      // known stderr prefixes.
      if (std::getenv("BCC32_PROBE_SUPPORT_DIAG") != nullptr) {
        std::uint32_t probe_bins[construction::kRelationProbeSupportBins] = {};
        std::uint32_t probe_total_host = 0u;
        cuda_require(cudaMemcpy(probe_bins,
                                state.relation_probe_support_histogram.get(),
                                sizeof(probe_bins), cudaMemcpyDeviceToHost),
                     "read relation probe support histogram");
        cuda_require(cudaMemcpy(&probe_total_host,
                                state.relation_probe_total.get(),
                                sizeof(probe_total_host),
                                cudaMemcpyDeviceToHost),
                     "read relation probe total");
        std::fprintf(stderr,
                     "probe_support total=%u b0=%u b1=%u b2=%u b3=%u b4=%u"
                     " b5=%u b6=%u b7plus=%u ge1=%u ge2=%u\n",
                     probe_total_host, probe_bins[0], probe_bins[1],
                     probe_bins[2], probe_bins[3], probe_bins[4],
                     probe_bins[5], probe_bins[6], probe_bins[7],
                     probe_total_host - probe_bins[0],
                     probe_total_host - probe_bins[0] - probe_bins[1]);
        // The ungated arm of the same measurement, when it was collected.
        if (std::getenv("BCC32_SHADOW_CENSUS") != nullptr) {
          std::uint32_t census_bins[construction::kRelationProbeSupportBins] =
              {};
          std::uint32_t census_total_host = 0u;
          std::uint32_t census_singletons_host = 0u;
          cuda_require(cudaMemcpy(census_bins,
                                  state.relation_census_histogram.get(),
                                  sizeof(census_bins), cudaMemcpyDeviceToHost),
                       "read relation census support histogram");
          cuda_require(cudaMemcpy(&census_total_host,
                                  state.relation_census_total.get(),
                                  sizeof(census_total_host),
                                  cudaMemcpyDeviceToHost),
                       "read relation census total");
          cuda_require(cudaMemcpy(&census_singletons_host,
                                  state.relation_census_source_singletons.get(),
                                  sizeof(census_singletons_host),
                                  cudaMemcpyDeviceToHost),
                       "read relation census source singleton total");
          std::fprintf(stderr,
                       "shadow_census total=%u b0=%u b1=%u b2=%u b3=%u b4=%u"
                       " b5=%u b6=%u b7plus=%u ge1=%u ge2=%u"
                       " source_singletons=%u\n",
                       census_total_host, census_bins[0], census_bins[1],
                       census_bins[2], census_bins[3], census_bins[4],
                       census_bins[5], census_bins[6], census_bins[7],
                       census_total_host - census_bins[0],
                       census_total_host - census_bins[0] - census_bins[1],
                       census_singletons_host);
          // The matched-counterfactual arm, on its own line so the census
          // format above is unchanged for existing readers. This is the arm
          // that decides whether the topic is doing any work: if the topic
          // and a matched alternative subject find support at the same rate,
          // the store is an episodic membership index, not a relation model.
          std::uint32_t cf_bins[construction::kRelationProbeSupportBins] = {};
          std::uint32_t cf_total_host = 0u;
          std::uint32_t topic_greater_host = 0u;
          std::uint32_t cf_greater_host = 0u;
          std::uint32_t ties_host = 0u;
          cuda_require(
              cudaMemcpy(cf_bins,
                         state.relation_counterfactual_histogram.get(),
                         sizeof(cf_bins), cudaMemcpyDeviceToHost),
              "read relation counterfactual histogram");
          cuda_require(cudaMemcpy(&cf_total_host,
                                  state.relation_counterfactual_total.get(),
                                  sizeof(cf_total_host),
                                  cudaMemcpyDeviceToHost),
                       "read relation counterfactual total");
          cuda_require(cudaMemcpy(&topic_greater_host,
                                  state.relation_topic_strictly_greater.get(),
                                  sizeof(topic_greater_host),
                                  cudaMemcpyDeviceToHost),
                       "read relation topic strictly greater total");
          cuda_require(
              cudaMemcpy(&cf_greater_host,
                         state.relation_counterfactual_strictly_greater.get(),
                         sizeof(cf_greater_host), cudaMemcpyDeviceToHost),
              "read relation counterfactual strictly greater total");
          cuda_require(cudaMemcpy(&ties_host,
                                  state.relation_support_ties.get(),
                                  sizeof(ties_host), cudaMemcpyDeviceToHost),
                       "read relation support tie total");
          std::fprintf(stderr,
                       "shadow_counterfactual total=%u b0=%u ge1=%u ge2=%u"
                       " topic_greater=%u counterfactual_greater=%u ties=%u\n",
                       cf_total_host, cf_bins[0],
                       cf_total_host - cf_bins[0],
                       cf_total_host - cf_bins[0] - cf_bins[1],
                       topic_greater_host, cf_greater_host, ties_host);
        }
      }
      // Persist the discourse state for future replies (cued or quiet): the
      // newest active theme continues the monologue, and everything said so
      // far stays refractory so the organism advances instead of restating.
      if (sentences != 0u) {
        state.discourse_theme =
            !front.empty() ? front.back() : state.discourse_theme;
        // Persist only the topics a CUED reply was ANCHORED on -- the
        // attested first theme (the cue's own topic) and the first
        // continuation theme. Persisting every theme (or every rheme) let
        // one reply's tail chain evict the standing conversation topics
        // and drift the quiet monologue off the contact vocabulary. Quiet
        // musing itself never writes the standing front: the
        // conversation's topics stay the anchor, and successive quiet
        // ticks keep returning to them (the per-tick chain still advances
        // through discourse_theme and the persisted refractory set below).
        if (!quiet_continuation) {
          // Push the continuation theme first and the ATTESTED cue topic
          // LAST: quiet continuation walks the standing front newest-first,
          // and the attested topic (the very words the interlocutor used)
          // is the stronger anchor to the exchange just held, so it must
          // be the one tried first.
          const std::size_t anchor_count =
              uttered_themes.size() < 2u ? uttered_themes.size() : 2u;
          for (std::size_t k = anchor_count; k-- != 0u;) {
            const std::uint32_t member = uttered_themes[k];
            bool known = false;
            for (const std::uint32_t prior : state.discourse_front)
              known |= prior == member;
            if (!known) state.discourse_front.push_back(member);
          }
          while (state.discourse_front.size() > 16u)
            state.discourse_front.erase(state.discourse_front.begin());
        }
        for (const std::uint32_t said : said_ends) {
          // Refresh recency for an already-known end: a value restated
          // across replies must move to the BACK, or it ages out of the
          // newest-half refractory seeding window while still being
          // re-said (the cross-tick restatement failure).
          for (auto it = state.discourse_said.begin();
               it != state.discourse_said.end(); ++it) {
            if (*it == said) {
              state.discourse_said.erase(it);
              break;
            }
          }
          state.discourse_said.push_back(said);
        }
        while (state.discourse_said.size() > kProgressionEndCap)
          state.discourse_said.erase(state.discourse_said.begin());
      }
      // A quiet continuation with too little new to say is not worth
      // interrupting the older autonomous behavior for: fall back unless the
      // monologue actually advanced (>= 3 sentences).
      if (quiet_continuation && sentences < 3u) {
        std::fprintf(stderr,
                     "discourse_continuation_faded sentences=%u\n", sentences);
        return {};
      }
      if (!reply.empty()) {
        while (!reply.empty() && reply.back() == ' ') reply.pop_back();
        std::fprintf(stderr, "relation_triple_reply sentences=%u bytes=%zu\n",
                     sentences, reply.size());
        return reply;
      }
      if (quiet_continuation) return {};
    }
  }
  if (!commitment_active && !state.content_commit_lesioned &&
      state.content_commitment_units.get() != nullptr &&
      state.online_association_count != 0u) {
    // RELATIONAL commitment first: the anchor-conditioned DIRECTED
    // transition store (anchor, previous -> next) carries which transitions
    // the subject licenses -- a directed claim, where co-occurrence mass
    // only carries salience. When the store holds no attested clause for
    // the topic the kernel commits nothing and the co-occurrence plan below
    // stands, so relation-commit can only add structure, never regress.
    // BCC32_LESION_RELATION_COMMIT skips it entirely (exact co-occurrence
    // commitment behavior).
    bool relational_commit = false;
    const bool relation_commit_lesioned =
        std::getenv("BCC32_LESION_RELATION_COMMIT") != nullptr;
    std::uint32_t commitment_len = 0u;
    if (!relation_commit_lesioned &&
        state.online_conditioned_transition_count != 0u) {
      construction::form_relational_commitment_kernel<<<1u, 1u>>>(
          state.online_conditioned_transitions.get(),
          state.online_conditioned_transition_conductance.get(),
          state.online_conditioned_transition_count,
          state.unit_vitality.get(),
          state.subject_ids.get(), state.subject_weights.get(),
          state.subject_count.get(), kSubjectCap,
          state.construction_closed_class_mask.get(), state.unit_lengths.get(),
          state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
          state.construction_filler_terminal_mask.get(),
          state.content_commitment_units.get(),
          state.content_commitment_meta.get());
      cuda_require(cudaGetLastError(), "form resident relational commitment");
      cuda_require(cudaMemcpy(&commitment_len,
                              state.content_commitment_meta.get(),
                              sizeof(commitment_len), cudaMemcpyDeviceToHost),
                   "read resident relational commitment extent");
      relational_commit = commitment_len >= construction::kConstructionMinSlots;
    }
    if (!relational_commit) {
      // Commitment-specific association mass: only CONTENT subject units act
      // as sources (question glue radiates mass onto generic filler).
      // Separate buffer -- the pool's mass accumulation above stays
      // untouched.
      DeviceArray<unsigned long long> commitment_association_mass(
          std::max<std::uint32_t>(1u, state.unit_count));
      cuda_require(cudaMemset(commitment_association_mass.get(), 0,
                              commitment_association_mass.bytes()),
                   "clear commitment association mass");
      construction::accumulate_commitment_association_kernel<<<
          blocks_for(state.online_association_count), kBlock>>>(
          state.online_associations.get(), state.online_association_counts.get(),
          state.online_association_count, state.subject_ids.get(),
          state.subject_count.get(), kSubjectCap,
          state.construction_closed_class_mask.get(),
          commitment_association_mass.get());
      cuda_require(cudaGetLastError(), "accumulate commitment association mass");
      construction::form_content_commitment_kernel<<<1u, 1u>>>(
          commitment_association_mass.get(), state.unit_vitality.get(),
          state.unit_count,
          state.subject_ids.get(), state.subject_weights.get(),
          state.subject_count.get(), kSubjectCap,
          state.construction_closed_class_mask.get(), state.unit_lengths.get(),
          state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
          state.construction_filler_terminal_mask.get(),
          state.content_commitment_units.get(),
          state.content_commitment_meta.get());
      cuda_require(cudaGetLastError(), "form resident content commitment");
      cuda_require(cudaMemcpy(&commitment_len,
                              state.content_commitment_meta.get(),
                              sizeof(commitment_len), cudaMemcpyDeviceToHost),
                   "read resident content commitment extent");
    }
    commitment_active = commitment_len >= construction::kConstructionMinSlots;
    std::fprintf(stderr, "commitment mode=%s len=%u active=%u",
                 relational_commit ? "relational" : "cooccurrence",
                 commitment_len, commitment_active ? 1u : 0u);
    // Diagnostics only: surface the committed words on stderr.
    std::vector<std::uint32_t> commitment_host(
        std::min<std::uint32_t>(commitment_len, construction::kCommitmentCap));
    if (!commitment_host.empty()) {
      cuda_require(cudaMemcpy(commitment_host.data(),
                              state.content_commitment_units.get(),
                              commitment_host.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read resident content commitment plan");
      for (const std::uint32_t unit : commitment_host) {
        std::uint32_t length = 0u;
        cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                                sizeof(length), cudaMemcpyDeviceToHost),
                     "read commitment trace unit length");
        std::uint32_t words[kUnitWords] = {};
        cuda_require(cudaMemcpy(words,
                                state.unit_content.get() +
                                    static_cast<std::size_t>(unit) * kUnitWords,
                                sizeof(words), cudaMemcpyDeviceToHost),
                     "read commitment trace unit bytes");
        std::fprintf(stderr, " '");
        const std::uint32_t bounded = std::min<std::uint32_t>(length, 24u);
        for (std::uint32_t offset = 0u; offset < bounded; ++offset) {
          const std::uint8_t byte = static_cast<std::uint8_t>(
              words[offset / 4u] >> ((offset % 4u) * 8u));
          std::fputc(byte >= 32u && byte < 127u ? byte : '.', stderr);
        }
        std::fprintf(stderr, "'");
      }
    }
    std::fprintf(stderr, "\n");
  }
  construction::build_construction_pool_kernel<<<1u, 1u>>>(
      state.subject_ids.get(), state.subject_weights.get(),
      state.subject_count.get(), kSubjectCap, state.motor_context.get(),
      state.motor_completion.get(), kCompositionUnits,
      state.construction_roles.get(), state.construction_closed_class_mask.get(),
      state.unit_lengths.get(),
      state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
      state.construction_closure_bytes.get(),
      state.construction_closure_count, allow_completion,
      8u * kSubjectInitWeight, construction_association_mass.get(),
      state.unit_count, 48u, state.construction_filler_terminal_mask.get(),
      state.construction_pool_units.get(),
      state.construction_pool_roles.get(), state.construction_pool_weights.get(),
      state.construction_pool_meta.get());
  cuda_require(cudaGetLastError(), "gather resident construction pool");
  cuda_require(cudaMemset(state.construction_best.get(), 0,
                          state.construction_best.bytes()),
               "clear resident construction choice");
  construction::select_construction_kernel<<<
      (state.construction_count_host + construction::kConstructionBlock - 1u) /
          construction::kConstructionBlock,
      construction::kConstructionBlock>>>(
      state.construction_tokens.get(), state.construction_lengths.get(),
      state.construction_slot_counts.get(), state.construction_supports.get(),
      state.construction_store_count.get(), state.construction_pool_units.get(),
      state.construction_pool_roles.get(),
      state.construction_pool_weights.get(), state.construction_pool_meta.get(),
      state.construction_roles.get(),
      commitment_active ? state.content_commitment_units.get() : nullptr,
      commitment_active ? state.content_commitment_meta.get() : nullptr,
      state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
      state.boundary_mask.get(),
      state.morph_agreement_lesioned
          ? nullptr
          : state.construction_suffix_transitions.get(),
      state.construction_last_selected.get(),
      state.construction_best.get());
  cuda_require(cudaGetLastError(), "select resident construction");
  construction::bind_construction_kernel<BigramKey><<<1u, 1u>>>(
      state.construction_tokens.get(), state.construction_lengths.get(),
      state.construction_slot_counts.get(), state.construction_store_count.get(),
      state.construction_best.get(),
      state.construction_pool_units.get(), state.construction_pool_roles.get(),
      state.construction_pool_weights.get(), state.construction_pool_meta.get(),
      state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
      state.construction_initial_form_mask.get(),
      state.bigrams.get(), state.bigram_counts.get(), state.bigram_count,
      state.boundary_mask.get(),
      state.morph_agreement_lesioned
          ? nullptr
          : state.construction_suffix_transitions.get(),
      commitment_active ? state.content_commitment_units.get() : nullptr,
      commitment_active ? state.content_commitment_meta.get() : nullptr,
      state.construction_last_selected.get(),
      state.construction_plan.get(), state.construction_plan_meta.get());
  cuda_require(cudaGetLastError(), "bind resident construction slots");
  std::uint32_t plan_meta[3] = {};
  cuda_require(cudaMemcpy(plan_meta, state.construction_plan_meta.get(),
                          sizeof(plan_meta), cudaMemcpyDeviceToHost),
               "read resident construction plan");
  if (plan_meta[0] == 0u) {
    std::uint32_t pool_probe = 0u;
    cuda_require(cudaMemcpy(&pool_probe, state.construction_pool_meta.get(),
                            sizeof(pool_probe), cudaMemcpyDeviceToHost),
                 "read resident construction pool extent");
    std::fprintf(stderr,
                 "construction_skip pool=%u subjects=%u store=%u\n",
                 pool_probe, subject_field_count, state.construction_count_host);
    return {};
  }
  std::vector<std::uint32_t> skeleton(plan_meta[0]);
  std::vector<std::uint32_t> bound_plan(plan_meta[0]);
  cuda_require(cudaMemcpy(
                   skeleton.data(),
                   state.construction_tokens.get() +
                       static_cast<std::size_t>(plan_meta[1]) *
                           construction::kConstructionMaxTokens,
                   skeleton.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read selected resident construction skeleton");
  cuda_require(cudaMemcpy(bound_plan.data(), state.construction_plan.get(),
                          bound_plan.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "read bound resident construction plan");
  DeviceArray<std::uint8_t> device_output(output_bytes);
  DeviceArray<std::uint32_t> device_generated_count(1u);
  construction::emit_construction_plan_kernel<<<1u, 1u>>>(
      state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
      state.construction_plan.get(), state.construction_plan_meta.get(),
      state.boundary_mask.get(), device_output.get(), output_bytes,
      device_generated_count.get());
  cuda_require(cudaGetLastError(), "realize resident construction");
  cuda_require(cudaDeviceSynchronize(), "complete resident construction reply");
  std::uint32_t generated_count = 0u;
  cuda_require(cudaMemcpy(&generated_count, device_generated_count.get(),
                          sizeof(generated_count), cudaMemcpyDeviceToHost),
               "read resident construction extent");
  if (generated_count == 0u || generated_count > output_bytes) return {};
  std::vector<std::uint8_t> output(generated_count);
  cuda_require(cudaMemcpy(output.data(), device_output.get(), generated_count,
                          cudaMemcpyDeviceToHost),
               "read resident construction bytes");
  std::fprintf(stderr, "construction_reply construction=%u tokens=%u slots=%u\n",
               plan_meta[1], plan_meta[0], plan_meta[2]);
  // Per-slot candidate-pool sizes (the role-canon capability metric): how
  // many active pool units carry each slot's demanded role.
  {
    std::vector<std::uint32_t> pool_meta_host(1u + roles::kStructuralRoleCount);
    cuda_require(cudaMemcpy(pool_meta_host.data(),
                            state.construction_pool_meta.get(),
                            pool_meta_host.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read resident construction pool histogram");
    std::fprintf(stderr, "construction_slot_pools pool=%u", pool_meta_host[0]);
    for (std::uint32_t i = 0u; i < plan_meta[0]; ++i) {
      if (!construction::token_is_slot(skeleton[i])) continue;
      const std::uint32_t role = construction::token_role(skeleton[i]);
      std::fprintf(stderr, " s%u:r%u=%u", i, role,
                   role < roles::kStructuralRoleCount ? pool_meta_host[1u + role]
                                                      : 0u);
    }
    std::fprintf(stderr, "\n");
  }
  std::fprintf(stderr, "construction_trace");
  for (std::uint32_t i = 0u; i < plan_meta[0]; ++i) {
    std::fprintf(stderr, " %c%u->u%u",
                 construction::token_is_slot(skeleton[i]) ? 'S' : 'L',
                 construction::token_is_slot(skeleton[i])
                     ? construction::token_role(skeleton[i])
                     : skeleton[i],
                 bound_plan[i]);
  }
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "construction_trace_bytes");
  for (std::uint32_t i = 0u; i < plan_meta[0]; ++i) {
    const std::uint32_t unit = bound_plan[i];
    roles::MutableStructuralRole role{};
    std::uint32_t vitality = 0u;
    std::uint32_t role_population = 0u;
    cuda_require(cudaMemcpy(&role, state.construction_roles.get() + unit,
                            sizeof(role), cudaMemcpyDeviceToHost),
                 "read construction trace role");
    cuda_require(cudaMemcpy(&vitality, state.unit_vitality.get() + unit,
                            sizeof(vitality), cudaMemcpyDeviceToHost),
                 "read construction trace vitality");
    cuda_require(cudaMemcpy(&role_population,
                            state.construction_role_population.get() + role.role,
                            sizeof(role_population), cudaMemcpyDeviceToHost),
                 "read construction trace role population");
    std::uint32_t length = 0u;
    cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                            sizeof(length), cudaMemcpyDeviceToHost),
                 "read construction trace unit length");
    const std::uint32_t bounded_length = std::min<std::uint32_t>(length, 24u);
    std::uint32_t words[kUnitWords] = {};
    cuda_require(cudaMemcpy(words,
                            state.unit_content.get() +
                                static_cast<std::size_t>(unit) * kUnitWords,
                            sizeof(words), cudaMemcpyDeviceToHost),
                 "read construction trace unit bytes");
    std::fprintf(stderr, " %c%u[r%u,c%u,e%u,v%u,p%u]:'",
                 construction::token_is_slot(skeleton[i]) ? 'S' : 'L',
                 construction::token_is_slot(skeleton[i])
                     ? construction::token_role(skeleton[i])
                     : skeleton[i],
                 role.role, role.confidence, role.evidence_depth, vitality,
                 role_population);
    for (std::uint32_t offset = 0u; offset < bounded_length; ++offset) {
      const std::uint8_t byte = static_cast<std::uint8_t>(
          words[offset / 4u] >> ((offset % 4u) * 8u));
      std::fputc(byte >= 32u && byte < 127u ? byte : '.', stderr);
    }
    std::fprintf(stderr, "'");
  }
  std::fprintf(stderr, "\n");
  return output;
}
