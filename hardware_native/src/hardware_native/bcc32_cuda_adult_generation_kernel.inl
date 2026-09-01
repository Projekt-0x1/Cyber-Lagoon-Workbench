__global__ void generate_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* vitality, const std::uint32_t* top_ids,
    const BigramKey* bigrams, const std::uint32_t* bigram_counts, std::uint32_t bigram_count,
    const TrigramKey* trigrams, const std::uint32_t* trigram_counts, std::uint32_t trigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count, const TrigramKey* online_trigrams,
    const std::uint32_t* online_trigram_counts, std::uint32_t online_trigram_count,
    const ConditionedTransitionKey* online_cond,
    const std::uint32_t* online_cond_counts, std::uint32_t online_cond_count,
    std::uint32_t* motor_context, std::uint32_t* motor_completion,
    const std::uint32_t* subject_ids, const std::uint32_t* subject_weights,
    std::uint32_t subject_count,
    const std::uint32_t* qonset_count, const std::uint32_t* qterm_count,
    const std::uint32_t* qorig_onset, const std::uint32_t* qorig_onset_w,
    const std::uint32_t* qorig_onset_n, const std::uint32_t* qorig_term,
    const std::uint32_t* qorig_term_w, const std::uint32_t* qorig_term_n,
    std::uint32_t qorig_on,
    std::uint32_t* resident_rng, std::uint8_t* output, std::uint32_t output_bytes,
    std::uint32_t* generated_count) {
  __shared__ std::uint32_t candidate_ids[kBlock * kTopK];
  __shared__ std::uint32_t candidate_weights[kBlock * kTopK];
  __shared__ std::uint32_t global_ids[kTopK];
  __shared__ std::uint32_t global_weights[kTopK];
  __shared__ std::uint32_t rng;
  __shared__ std::uint32_t first;
  __shared__ std::uint32_t second;
  __shared__ std::uint32_t selected;
  __shared__ std::uint32_t written;
  __shared__ std::uint32_t emitted_units;
  __shared__ std::uint32_t range_begin;
  __shared__ std::uint32_t range_end;
  __shared__ std::uint32_t found;
  __shared__ std::uint32_t online_begin;
  __shared__ std::uint32_t online_end;
  __shared__ std::uint32_t subj_ids_s[kSubjectCap];
  __shared__ std::uint32_t subj_w_s[kSubjectCap];
  __shared__ std::uint32_t subj_n;
  __shared__ std::uint32_t drift_run;
  __shared__ std::uint32_t continue_walk;
  __shared__ std::uint32_t recent_units[kMotorRecentUnits];
  __shared__ std::uint32_t recent_count;
  __shared__ std::uint32_t recent_head;
  __shared__ std::uint32_t resident_source_run;
  __shared__ std::uint32_t resident_episode_next;
  __shared__ std::uint32_t ctx_hist_first[kCtxHistWindow];
  __shared__ std::uint32_t ctx_hist_second[kCtxHistWindow];
  __shared__ std::uint32_t ctx_hist_n;
  __shared__ std::uint32_t forced_run;
  __shared__ std::uint32_t candidate_diversity;
  // QUESTION-ORIGINATION walk state (--qorig, off by default -- see the
  // kernel-body guards below; when qorig_on==0 these are set once and never
  // touched again, zero behavior change). q_drive is the resident predictive
  // uncertainty of the just-emitted unit (0..256, high when its occurrence
  // count is low = a knowledge gap); q_prev_end gates the onset jump to
  // sentence boundaries; q_in/q_len track an opened question span so the
  // terminal bias/forced-close only apply INSIDE one.
  __shared__ std::uint32_t q_drive;
  __shared__ std::uint32_t q_len;
  __shared__ std::uint32_t q_in;
  __shared__ std::uint32_t q_prev_end;
  __shared__ std::uint32_t autonomous_generation;

  if (motor_context[5] == 2u) {
    if (threadIdx.x == 0u) generated_count[0] = 0u;
    return;
  }

  // Load the resident subject field into shared memory (a local, decaying copy
  // so multi-seed generations start from the same activation). subj_n caps at
  // kSubjectCap. The field is only active when the cue conditioned it.
  if (threadIdx.x == 0u) {
    autonomous_generation = motor_context[5] == 0u ? 1u : 0u;
    if (autonomous_generation != 0u) motor_context[3] = 0u;
    subj_n = subject_count < kSubjectCap ? subject_count : kSubjectCap;
    drift_run = 0u;
    continue_walk = 0u;
    recent_count = 0u;
    recent_head = 0u;
    resident_source_run = 0u;
    resident_episode_next = 0xffffffffu;
    ctx_hist_n = 0u;
    forced_run = 0u;
    q_drive = 0u;
    q_len = 0u;
    q_in = 0u;
    q_prev_end = 1u;  // the start of generation counts as a sentence boundary
  }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < subj_n; i += blockDim.x) {
    subj_ids_s[i] = subject_ids[i];
    subj_w_s[i] = subject_weights[i];
  }
  __syncthreads();
  if (motor_context[5] == 1u || motor_context[5] == 3u || motor_context[5] == 4u) {
    if (threadIdx.x == 0u) written = 0u;
    __syncthreads();
    for (std::uint32_t item = 0u; item < motor_context[3] && written < output_bytes; ++item) {
      const std::uint32_t unit = motor_completion[item];
      const std::uint32_t length = min(unit_lengths[unit], output_bytes - written);
      for (std::uint32_t offset = threadIdx.x; offset < length; offset += blockDim.x) {
        const std::uint32_t packed = unit_content[unit * kUnitWords + offset / 4u];
        output[written + offset] =
            static_cast<std::uint8_t>(packed >> ((offset % 4u) * 8u));
      }
      __syncthreads();
      if (threadIdx.x == 0u) written += length;
      __syncthreads();
    }
    if (threadIdx.x == 0u) {
      if (motor_context[0] != 0u && subj_n > 0u && written < output_bytes) {
        // Re-enter the subject field after the composed answer: seed the walk
        // from the composed tail and force an immediate re-anchor so the
        // continuation lands back on the question's subject instead of stopping
        // at the short (drifting) composed span.
        rng = resident_rng[0];
        const std::uint32_t cnt = motor_context[3];
        if (cnt >= 2u) {
          first = motor_completion[cnt - 2u];
          second = motor_completion[cnt - 1u];
          emitted_units = 2u;
        } else if (cnt == 1u) {
          first = motor_completion[0];
          second = first;
          emitted_units = 1u;
        } else {
          first = motor_context[1];
          second = first;
          emitted_units = 1u;
        }
        drift_run = kSubjectDriftUnits;
        continue_walk = 1u;
      }
    }
    __syncthreads();
    if (continue_walk == 0u) {
      if (threadIdx.x == 0u) generated_count[0] = written;
      return;
    }
  }

  if (continue_walk == 0u && threadIdx.x == 0u) {
    rng = resident_rng[0];
    first = motor_context[0] != 0u ? motor_context[1]
                                    : choose_unigram(top_ids, vitality, &rng);
    second = motor_context[5] == 6u && motor_context[7] >= 2u
        ? motor_context[6] : first;
    written = 0u;
    emitted_units = motor_context[5] == 6u ? motor_context[7] : 0u;
  }
  __syncthreads();

  while (written < output_bytes) {
    if (threadIdx.x == 0u) {
      selected = first;
      found = 0u;
      online_begin = online_end = 0u;
      if (emitted_units >= 2u) {
        online_begin = lower_trigram(online_trigrams, online_trigram_count, first, second);
        online_end = upper_trigram(online_trigrams, online_trigram_count, first, second);
      }
      range_begin = emitted_units >= 2u
          ? lower_trigram(trigrams, trigram_count, first, second) : 0u;
      range_end = emitted_units >= 2u
          ? upper_trigram(trigrams, trigram_count, first, second) : 0u;
    }
    for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
      candidate_ids[threadIdx.x * kTopK + slot] = 0u;
      candidate_weights[threadIdx.x * kTopK + slot] = 0u;
    }
    __syncthreads();

    if (emitted_units != 0u && emitted_units >= 2u) {
      std::uint32_t local_ids[kTopK] = {};
      std::uint32_t local_weights[kTopK] = {};
      for (std::uint32_t i = range_begin + threadIdx.x; i < range_end; i += blockDim.x) {
        if (trigram_counts[i] != 0u) {
          insert_top(trigrams[i].next, 5u * evidence_depth(trigram_counts[i]),
                     local_ids, local_weights, kTopK);
        }
      }
      for (std::uint32_t i = online_begin + threadIdx.x; i < online_end; i += blockDim.x) {
        if (online_trigram_counts[i] != 0u && online_trigrams[i].first == first &&
            online_trigrams[i].second == second) {
          insert_top(online_trigrams[i].next, 4u * evidence_depth(online_trigram_counts[i]),
                     local_ids, local_weights, kTopK);
        }
      }
      for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
        candidate_ids[threadIdx.x * kTopK + slot] = local_ids[slot];
        candidate_weights[threadIdx.x * kTopK + slot] = local_weights[slot];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0u && emitted_units != 0u) {
      for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
        global_ids[slot] = 0u;
        global_weights[slot] = 0u;
      }
      for (std::uint32_t i = 0u; i < kBlock * kTopK; ++i) {
        if (candidate_weights[i] != 0u) {
          insert_top(candidate_ids[i], candidate_weights[i], global_ids, global_weights, kTopK);
          found = 1u;
        }
      }
      if (found == 0u) {
        const std::uint32_t context = emitted_units >= 2u ? second : first;
        online_begin = lower_bigram(online_bigrams, online_bigram_count, context);
        online_end = upper_bigram(online_bigrams, online_bigram_count, context);
        range_begin = lower_bigram(bigrams, bigram_count, context);
        range_end = upper_bigram(bigrams, bigram_count, context);
      }
    }
    __syncthreads();

    if (emitted_units != 0u && found == 0u) {
      std::uint32_t local_ids[kTopK] = {};
      std::uint32_t local_weights[kTopK] = {};
      for (std::uint32_t i = range_begin + threadIdx.x; i < range_end; i += blockDim.x) {
        if (bigram_counts[i] != 0u) {
          insert_top(bigrams[i].next, 5u * evidence_depth(bigram_counts[i]),
                     local_ids, local_weights, kTopK);
        }
      }
      const std::uint32_t context = emitted_units >= 2u ? second : first;
      for (std::uint32_t i = online_begin + threadIdx.x; i < online_end; i += blockDim.x) {
        if (online_bigram_counts[i] != 0u && online_bigrams[i].previous == context) {
          insert_top(online_bigrams[i].next, 4u * evidence_depth(online_bigram_counts[i]),
                     local_ids, local_weights, kTopK);
        }
      }
      for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
        candidate_ids[threadIdx.x * kTopK + slot] = local_ids[slot];
        candidate_weights[threadIdx.x * kTopK + slot] = local_weights[slot];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0u && emitted_units != 0u) {
      if (found == 0u) {
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          global_ids[slot] = 0u;
          global_weights[slot] = 0u;
        }
        for (std::uint32_t i = 0u; i < kBlock * kTopK; ++i) {
          if (candidate_weights[i] != 0u) {
            insert_top(candidate_ids[i], candidate_weights[i],
                       global_ids, global_weights, kTopK);
            found = 1u;
          }
        }
      }
      // Subject-field BIAS: score up any candidate transition that lands on an
      // active subject unit so the on-topic draft wins over the generic replay.
      if (found != 0u && subj_n > 0u) {
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          if (global_weights[slot] == 0u) continue;
          for (std::uint32_t k = 0u; k < subj_n; ++k) {
            if (subj_ids_s[k] == global_ids[slot]) {
              global_weights[slot] += subj_w_s[k];
              break;
            }
          }
        }
      }
      // Subject-CONDITIONED-transition bias: additionally boost trigram
      // candidates that are attested continuations of `second` when an active
      // subject anchor was recently present (online_conditioned_transitions).
      // Unlike the unit bias above (which only rewards landing ON a subject
      // word), this rewards the transitions the subject actually licenses --
      // steering the order-2 walk toward subject-specific grammatical paths.
      // Candidates stay within the trigram set, so local (order-2) coherence is
      // preserved; only the subject-appropriate ones are lifted.
      if (found != 0u && subj_n > 0u && online_cond_count > 0u &&
          emitted_units >= 1u) {
        const std::uint32_t prev = second;
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          if (global_weights[slot] == 0u) continue;
          const std::uint32_t cand = global_ids[slot];
          std::uint32_t boost = 0u;
          for (std::uint32_t k = 0u; k < subj_n; ++k) {
            const std::uint32_t c =
                bcc32_cuda_resident_synthesis::synthesis_subject_transition_count(
                    online_cond, online_cond_counts, online_cond_count,
                    subj_ids_s[k], prev, cand);
            if (c != 0u) boost += evidence_depth(c) * evidence_depth(subj_w_s[k]);
          }
          global_weights[slot] += boost;
        }
      }
      if (found != 0u && recent_count != 0u) {
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          if (global_weights[slot] == 0u) continue;
          std::uint32_t repeats = 0u;
          for (std::uint32_t prior = 0u; prior < recent_count; ++prior) {
            repeats += recent_units[prior] == global_ids[slot];
          }
          if (repeats != 0u) {
            const std::uint32_t shift = min(6u, 2u * repeats);
            global_weights[slot] = max(1u, global_weights[slot] >> shift);
          }
        }
      }
      resident_episode_next = 0xffffffffu;
      if (motor_context[5] == 6u && emitted_units >= 2u &&
          resident_source_run < kCompositionSourceRunLimit) {
        for (std::uint32_t position = 2u; position < motor_context[3]; ++position) {
          if (motor_completion[position - 2u] == first &&
              motor_completion[position - 1u] == second) {
            resident_episode_next = motor_completion[position];
            insert_top(resident_episode_next, 65536u, global_ids, global_weights, kTopK);
          }
        }
      }
      // QUESTION-TERMINAL bias (--qorig): while inside an originated question,
      // boost any candidate this walk already attests as a resident '?'-fused
      // unit (qterm_count > 0), so the ordinary weighted draw below naturally
      // gravitates toward a legal, corpus-attested close instead of needing a
      // separate question-conditioned transition table -- reuses the existing
      // order-2 candidate set exactly as the subject bias above does.
      if (qorig_on != 0u && qterm_count != nullptr && found != 0u && q_in != 0u) {
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          if (global_weights[slot] == 0u) continue;
          const std::uint32_t tc = qterm_count[global_ids[slot]];
          if (tc != 0u) {
            global_weights[slot] += min(tc, kQOrigCap);
          }
        }
      }
      // Diversity check for the forced-run break below: count how many
      // distinct candidates carry any evidence AFTER biasing. found==1 with
      // candidate_diversity<=1 means no alternative continuation exists
      // anywhere in resident experience -- the walk is about to be forced,
      // not choosing.
      candidate_diversity = 0u;
      if (found != 0u) {
        for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
          if (global_weights[slot] != 0u) ++candidate_diversity;
        }
      }
      selected = found != 0u ? choose_weighted(&rng, global_ids, global_weights, kTopK)
                             : choose_unigram(top_ids, vitality, &rng);
      resident_source_run = selected == resident_episode_next
          ? resident_source_run + 1u : 0u;
      // RE-ANCHOR on drift: after kSubjectDriftUnits off-field units, rewrite the
      // selection to a resident subject unit (weighted by activation x length),
      // re-applying the strong topical lever mid-walk.
      if (subj_n > 0u && drift_run >= kSubjectDriftUnits) {
        // Grammar-aware re-anchor: prefer a subject unit that is ALSO a valid
        // transition continuation of the current context (present in the
        // candidate set with real evidence). Selecting it keeps recovery
        // grammatical instead of hard-injecting a context-free content word.
        // Only when grammar offers NO on-subject continuation do we fall back to
        // the context-free weighted jump (accepting one break to recover topic).
        std::uint32_t anchor_slot = kTopK;
        std::uint32_t anchor_w = 0u;
        if (found != 0u) {
          for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
            if (global_weights[slot] == 0u) continue;
            bool on_subj = false;
            for (std::uint32_t k = 0u; k < subj_n; ++k) {
              if (subj_ids_s[k] == global_ids[slot]) { on_subj = true; break; }
            }
            if (on_subj && global_weights[slot] > anchor_w) {
              anchor_w = global_weights[slot];
              anchor_slot = slot;
            }
          }
        }
        if (anchor_slot != kTopK) {
          selected = global_ids[anchor_slot];
          for (std::uint32_t k = 0u; k < subj_n; ++k) {
            if (subj_ids_s[k] == selected) {
              const std::uint32_t rw = subj_w_s[k] + kSubjectReinforce;
              subj_w_s[k] = rw > kSubjectCapWeight ? kSubjectCapWeight : rw;
              break;
            }
          }
          drift_run = 0u;
        } else {
          std::uint32_t tot = 0u;
          for (std::uint32_t k = 0u; k < subj_n; ++k) {
            const std::uint32_t L = unit_lengths[subj_ids_s[k]];
            const std::uint32_t lf = L >= 6u ? 3u : (L >= 4u ? 2u : 1u);
            tot += subj_w_s[k] * lf;
          }
          if (tot != 0u) {
            std::uint32_t x = rng;
            x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; rng = x;
            std::uint32_t draw = x % tot;
            std::uint32_t pick = 0u;
            for (std::uint32_t k = 0u; k < subj_n; ++k) {
              const std::uint32_t L = unit_lengths[subj_ids_s[k]];
              const std::uint32_t lf = L >= 6u ? 3u : (L >= 4u ? 2u : 1u);
              const std::uint32_t w = subj_w_s[k] * lf;
              if (draw < w) { pick = k; break; }
              draw -= w;
            }
            selected = subj_ids_s[pick];
            const std::uint32_t rw = subj_w_s[pick] + kSubjectReinforce;
            subj_w_s[pick] = rw > kSubjectCapWeight ? kSubjectCapWeight : rw;
            drift_run = 0u;
          }
        }
      } else if (subj_n > 0u) {
        // REINFORCE / DECAY: reinforce the field when the walk re-emits an
        // on-subject unit (self-sustaining topic); otherwise count a drift step
        // and let the field decay slowly so it can shift gradually.
        int hit = -1;
        for (std::uint32_t k = 0u; k < subj_n; ++k) {
          if (subj_ids_s[k] == selected) { hit = static_cast<int>(k); break; }
        }
        if (hit >= 0) {
          const std::uint32_t rw = subj_w_s[hit] + kSubjectReinforce;
          subj_w_s[hit] = rw > kSubjectCapWeight ? kSubjectCapWeight : rw;
          drift_run = 0u;
        } else {
          ++drift_run;
          for (std::uint32_t k = 0u; k < subj_n; ++k) {
            std::uint32_t w = subj_w_s[k];
            w -= (w >> 5);
            subj_w_s[k] = w < kSubjectFloorWeight ? kSubjectFloorWeight : w;
          }
        }
      }
      // CYCLE-BREAK (independent of subj_n -- runs whether or not a subject
      // field is active). If (first, second) is a context this generation has
      // already been in, the deterministic/near-deterministic walk is about to
      // retrace a short cycle (e.g. a strong, low-diversity trigram edge with
      // no competing candidates). Force a random unigram jump instead of
      // repeating the same continuation forever.
      if (emitted_units >= 2u) {
        bool revisited = false;
        for (std::uint32_t k = 0u; k < ctx_hist_n; ++k) {
          if (ctx_hist_first[k] == first && ctx_hist_second[k] == second) { revisited = true; break; }
        }
        if (revisited) selected = choose_unigram(top_ids, vitality, &rng);
        if (ctx_hist_n < kCtxHistWindow) {
          ctx_hist_first[ctx_hist_n] = first;
          ctx_hist_second[ctx_hist_n] = second;
          ++ctx_hist_n;
        } else {
          for (std::uint32_t k = 1u; k < kCtxHistWindow; ++k) {
            ctx_hist_first[k - 1u] = ctx_hist_first[k];
            ctx_hist_second[k - 1u] = ctx_hist_second[k];
          }
          ctx_hist_first[kCtxHistWindow - 1u] = first;
          ctx_hist_second[kCtxHistWindow - 1u] = second;
        }
      }
      // FORCED-RUN BREAK: see kMaxForcedRun. A single-candidate step means
      // choose_weighted had no real alternative; count consecutive such
      // steps and break away before the walk finishes replaying a long
      // unique corpus phrase verbatim. Resets on any step that had a real
      // choice, so an occasional unavoidable single-candidate step (a
      // genuinely rare but short collocation) is not penalized.
      if (found != 0u && candidate_diversity <= 1u) {
        ++forced_run;
        if (forced_run > kMaxForcedRun) {
          selected = choose_unigram(top_ids, vitality, &rng);
          forced_run = 0u;
        }
      } else {
        forced_run = 0u;
      }
      // QUESTION-ORIGINATION forced close (--qorig): once an opened question's
      // body has run kQOrigSoftCap units without reaching a resident
      // '?'-terminal through the bias above, force the close now -- prefer the
      // best-scoring candidate already in this step's set that IS a '?'-fused
      // unit (a legal, resident-attested transition); only fall back to the
      // unconditioned injected terminal list when the walk offers none.
      if (qorig_on != 0u && qterm_count != nullptr && q_in != 0u && q_len >= kQOrigSoftCap) {
        std::uint32_t best_slot = kTopK;
        std::uint32_t best_w = 0u;
        if (found != 0u) {
          for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
            if (global_weights[slot] == 0u) continue;
            if (qterm_count[global_ids[slot]] == 0u) continue;
            if (global_weights[slot] > best_w) { best_w = global_weights[slot]; best_slot = slot; }
          }
        }
        if (best_slot != kTopK) {
          selected = global_ids[best_slot];
        } else if (qorig_term != nullptr) {
          const std::uint32_t ln = min(*qorig_term_n, kQOrigInject);
          if (ln > 0u) {
            std::uint64_t tot = 0ull;
            for (std::uint32_t u = 0u; u < ln; ++u) tot += qorig_term_w[u];
            std::uint32_t g = qorig_term[0];
            if (tot > 0ull) {
              std::uint32_t x = rng; x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; rng = x;
              std::uint64_t draw = static_cast<std::uint64_t>(x) % tot;
              for (std::uint32_t u = 0u; u < ln; ++u) {
                if (draw < qorig_term_w[u]) { g = qorig_term[u]; break; }
                draw -= qorig_term_w[u];
              }
            }
            selected = g;
          }
        }
      }
      // QUESTION-ORIGINATION jump (--qorig): at a sentence boundary, outside an
      // already-open question, ORIGINATE one by jumping to an injected onset
      // unit -- probability rising in q_drive (the resident predictive
      // uncertainty of the just-emitted unit), so origination concentrates on
      // rare/under-learned contexts (gap-directed) rather than firing at every
      // sentence end. The jump target is drawn only from qorig_onset -- units
      // the online corpus itself attested as question-openers -- so this
      // reweights resident-legal continuations, it does not invent one.
      if (qorig_on != 0u && qorig_onset != nullptr && q_in == 0u &&
          q_prev_end != 0u && q_drive != 0u) {
        const std::uint32_t ln = min(*qorig_onset_n, kQOrigInject);
        if (ln > 0u) {
          std::uint32_t x = rng; x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; rng = x;
          const std::uint32_t prob = (q_drive * kQOrigRate) >> 8u;
          if ((x % 256u) < prob) {
            std::uint64_t tot = 0ull;
            for (std::uint32_t u = 0u; u < ln; ++u) tot += qorig_onset_w[u];
            std::uint32_t g = qorig_onset[0];
            if (tot > 0ull) {
              std::uint32_t y = rng; y ^= y << 13u; y ^= y >> 17u; y ^= y << 5u; rng = y;
              std::uint64_t draw = static_cast<std::uint64_t>(y) % tot;
              for (std::uint32_t u = 0u; u < ln; ++u) {
                if (draw < qorig_onset_w[u]) { g = qorig_onset[u]; break; }
                draw -= qorig_onset_w[u];
              }
            }
            selected = g;
            q_in = 1u;
            q_len = 0u;
          }
        }
      }
    }
    __syncthreads();

    // Preserve the resident unit sequence that actually produced an
    // input-free emission.  The motor-completion buffer is already a bounded
    // resident workspace; using it here keeps action provenance tied to the
    // generator's selected units instead of trying to infer a segmentation
    // from emitted bytes after the fact.
    if (threadIdx.x == 0u && autonomous_generation != 0u &&
        emitted_units < kCompositionUnits) {
      motor_completion[emitted_units] = selected;
    }
    __syncthreads();

    const std::uint32_t length = min(unit_lengths[selected], output_bytes - written);
    for (std::uint32_t offset = threadIdx.x; offset < length; offset += blockDim.x) {
      const std::uint32_t packed = unit_content[selected * kUnitWords + offset / 4u];
      output[written + offset] = static_cast<std::uint8_t>(packed >> ((offset % 4u) * 8u));
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      written += length;
      if (emitted_units == 0u) {
        first = selected;
      } else {
        first = second;
        second = selected;
      }
      ++emitted_units;
      if (autonomous_generation != 0u)
        motor_context[3] = min(emitted_units, kCompositionUnits);
      recent_units[recent_head] = selected;
      recent_head = (recent_head + 1u) % kMotorRecentUnits;
      recent_count = min(kMotorRecentUnits, recent_count + 1u);
      // QUESTION-ORIGINATION per-step state update (--qorig): recompute
      // q_drive (resident predictive uncertainty of the unit just emitted,
      // from its own occurrence count -- a rare unit's continuation is
      // poorly predicted, a gap) and advance the onset/terminal latch from
      // the FINAL selected unit this step. All resident matter; the walk
      // never consults host text.
      if (qorig_on != 0u) {
        const std::uint32_t space = static_cast<std::uint32_t>(' ');
        const bool ends_q = resident_unit_terminal_byte(unit_lengths, unit_content, selected,
                                                         static_cast<std::uint32_t>('?'), space);
        const bool ends_sentence = ends_q ||
            resident_unit_terminal_byte(unit_lengths, unit_content, selected,
                                        static_cast<std::uint32_t>('.'), space) ||
            resident_unit_terminal_byte(unit_lengths, unit_content, selected,
                                        static_cast<std::uint32_t>('!'), space) ||
            resident_unit_terminal_byte(unit_lengths, unit_content, selected,
                                        static_cast<std::uint32_t>('\n'), space);
        if (q_in != 0u) {
          ++q_len;
          if (ends_q) q_in = 0u;  // natural close via a resident '?'-fused unit
        }
        q_prev_end = ends_sentence ? 1u : 0u;
        const std::uint32_t occ = vitality[selected] < 2u ? 2u : vitality[selected];
        std::uint32_t lg = 0u;
        for (std::uint32_t v = occ; v > 1u; v >>= 1u) ++lg;  // floor(log2(occ))
        std::uint32_t conf = lg * kQOrigSubjScale;
        if (conf > 256u) conf = 256u;
        q_drive = conf >= 256u ? 0u : (256u - conf);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    resident_rng[0] = rng;
    generated_count[0] = written;
  }
}
