// ---------------------------------------------------------------------------
// POOL: gather the currently active content matter (resident subject field +
// composed completion units + subject-associated resident matter) with roles
// and activations. Serial (tiny).
// pool_meta[0] = pool count; pool_meta[1..64] = per-role histogram.
// ---------------------------------------------------------------------------
static __global__ void build_construction_pool_kernel(
    const std::uint32_t* subject_ids, const std::uint32_t* subject_weights,
    const std::uint32_t* subject_count, std::uint32_t subject_cap,
    const std::uint32_t* motor_context, const std::uint32_t* motor_completion,
    std::uint32_t completion_cap,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes,
    std::uint32_t closure_count, bool allow_completion,
    std::uint32_t completion_weight, unsigned long long* association_mass,
    std::uint32_t unit_count, std::uint32_t association_limit,
    const std::uint32_t* filler_terminal_mask,
    std::uint32_t* pool_units, std::uint32_t* pool_roles,
    std::uint32_t* pool_weights, std::uint32_t* pool_meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t count = 0u;
  for (std::uint32_t role = 0u; role <= resident_roles::kStructuralRoleCount; ++role)
    pool_meta[role] = 0u;

  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  for (std::uint32_t i = 0u; i < subjects && count < kConstructionPoolCap; ++i) {
    const std::uint32_t unit = subject_ids[i];
    if (unit_lengths[unit] < kConstructionMinFillerBytes) continue;
    if (construction_closed_class(unit, closed_class_mask)) continue;
    // Mid-sentence fillers must end in a word-interior-dominant byte: this
    // excludes closure carriers ("word.") AND clause-punctuation variants
    // ("growth;", "talent,") by resident byte statistics, not authored sets.
    {
      const std::uint32_t terminal = construction_terminal_byte(
          unit_lengths, unit_content, unit_words, unit, boundary_mask);
      if (terminal >= 256u || filler_terminal_mask[terminal] == 0u) continue;
    }
    bool seen = false;
    for (std::uint32_t k = 0u; k < count; ++k) {
      if (pool_units[k] == unit) {
        pool_weights[k] += subject_weights[i];
        seen = true;
        break;
      }
    }
    if (seen) continue;
    pool_units[count] = unit;
    pool_roles[count] = construction_role(roles[unit]);
    pool_weights[count] = subject_weights[i];
    ++count;
  }

  // Composed direct-answer content (relation composition output) joins the
  // pool with a strong activation whenever a composed plan is staged.
  const std::uint32_t mode = motor_context[5];
  if (allow_completion || mode == 1u || mode == 3u || mode == 4u) {
    const std::uint32_t completion_count =
        motor_context[3] < completion_cap ? motor_context[3] : completion_cap;
    for (std::uint32_t i = 0u; i < completion_count && count < kConstructionPoolCap;
         ++i) {
      const std::uint32_t unit = motor_completion[i];
      if (unit_lengths[unit] < kConstructionMinFillerBytes) continue;
      if (construction_closed_class(unit, closed_class_mask)) continue;
      {
        const std::uint32_t terminal = construction_terminal_byte(
            unit_lengths, unit_content, unit_words, unit, boundary_mask);
        if (terminal >= 256u || filler_terminal_mask[terminal] == 0u) continue;
      }
      bool seen = false;
      for (std::uint32_t k = 0u; k < count; ++k) {
        if (pool_units[k] == unit) {
          pool_weights[k] += completion_weight;
          seen = true;
          break;
        }
      }
      if (seen) continue;
      pool_units[count] = unit;
      pool_roles[count] = construction_role(roles[unit]);
      pool_weights[count] = completion_weight;
      ++count;
    }
  }

  // Subject-associated resident matter: the strongest association partners
  // of the active subjects join the pool so answers can carry content beyond
  // echoing the question's own words.
  if (association_mass != nullptr) {
    for (std::uint32_t added = 0u;
         added < association_limit && count < kConstructionPoolCap; ++added) {
      unsigned long long best_mass = 0ull;
      std::uint32_t best_unit = kNoConstruction;
      for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
        const unsigned long long mass = association_mass[unit];
        if (mass <= best_mass) continue;
        if (unit_lengths[unit] < kConstructionMinFillerBytes) {
          association_mass[unit] = 0ull;
          continue;
        }
        if (construction_closed_class(unit, closed_class_mask)) {
          association_mass[unit] = 0ull;
          continue;
        }
        {
          const std::uint32_t terminal = construction_terminal_byte(
              unit_lengths, unit_content, unit_words, unit, boundary_mask);
          if (terminal >= 256u || filler_terminal_mask[terminal] == 0u) {
            association_mass[unit] = 0ull;
            continue;
          }
        }
        best_mass = mass;
        best_unit = unit;
      }
      if (best_unit == kNoConstruction || best_mass < 2ull) break;
      association_mass[best_unit] = 0ull;
      bool seen = false;
      for (std::uint32_t k = 0u; k < count; ++k) seen |= pool_units[k] == best_unit;
      if (seen) continue;
      const unsigned long long capped = best_mass > 2048ull ? 2048ull : best_mass;
      pool_units[count] = best_unit;
      pool_roles[count] = construction_role(roles[best_unit]);
      pool_weights[count] = 128u + static_cast<std::uint32_t>(capped);
      ++count;
    }
  }

  pool_meta[0] = count;
  for (std::uint32_t k = 0u; k < count; ++k) {
    const std::uint32_t role = pool_roles[k];
    if (role < kConstructionRoleCount) ++pool_meta[1u + role];
  }
}

// ---------------------------------------------------------------------------
// SELECT: grammatical fit x generality. One thread per skeleton. The winner
// is the skeleton whose slot-role multiset is covered by the pool's role
// histogram, scored by support depth x matching activation mass. The query
// byte string plays no part here.
// ---------------------------------------------------------------------------
static __global__ void select_construction_kernel(
    const std::uint32_t* tokens, const std::uint32_t* lengths,
    const std::uint32_t* slot_counts, const std::uint32_t* supports,
    const std::uint32_t* construction_count,
    const std::uint32_t* pool_units, const std::uint32_t* pool_roles,
    const std::uint32_t* pool_weights,
    const std::uint32_t* pool_meta,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* commitment_units, const std::uint32_t* commitment_meta,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* suffix_transitions,
    const std::uint32_t* last_selected, unsigned long long* best_packed) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total =
      construction_count[0] < kConstructionCap ? construction_count[0] : kConstructionCap;
  if (index >= total) return;
  const std::uint32_t extent = lengths[index];
  if (extent == 0u) return;

  // RESIDENT VARIETY SEED. The jitter that separates near-tied frames is a
  // function of the organism's OWN prior choice and its OWN store size, both
  // device-resident. It was a host-incremented counter; Section 15 forbids an
  // external RNG or host causal policy in the resident path, and "it is only
  // jitter" is no defence -- jitter decides exactly the cases the organism's
  // evidence leaves tied. Identical resident state must now select identically
  // however many times the host has launched selection.
  const std::uint32_t resident_variety =
      construction_mix(last_selected[0] ^ (construction_count[0] * 0x9e3779b9u));

  // CONTENT-COMMITMENT MODE: the reply's content is already decided (an
  // ordered resident plan). A frame is now pure connective tissue: it is
  // scored by how well its literal junctions accept the committed units IN
  // ORDER (learned suffix-class agreement) plus slot-role fit -- never by
  // what content it could recruit from the pool. Frames demanding more
  // content than remains committed are rejected outright (replan/close is
  // the commitment's business, not the frame's).
  if (commitment_units != nullptr && commitment_meta != nullptr &&
      commitment_meta[0] > commitment_meta[1]) {
    const std::uint32_t cursor = commitment_meta[1];
    const std::uint32_t remaining = commitment_meta[0] - cursor;
    const std::uint32_t slots = slot_counts[index];
    if (slots == 0u || slots > remaining) return;
    std::uint32_t slot_index = 0u;
    std::uint32_t commit_agreement_deficit = 0u;
    std::uint32_t role_matches = 0u;
    for (std::uint32_t i = 0u; i < extent; ++i) {
      const std::uint32_t token = tokens[index * kConstructionMaxTokens + i];
      if (!token_is_slot(token)) continue;
      const std::uint32_t unit = commitment_units[cursor + slot_index];
      ++slot_index;
      if (roles != nullptr && roles[unit].confidence != 0u &&
          roles[unit].role == token_role(token))
        ++role_matches;
      if (suffix_transitions == nullptr) continue;
      const std::uint32_t unit_class = construction_suffix_class(
          unit_lengths, unit_content, unit_words, unit, boundary_mask);
      if (i != 0u) {
        const std::uint32_t previous =
            tokens[index * kConstructionMaxTokens + i - 1u];
        if (!token_is_slot(previous)) {
          const std::uint32_t left_class = construction_suffix_class(
              unit_lengths, unit_content, unit_words, previous, boundary_mask);
          if (left_class < kSuffixClassCount &&
              construction_suffix_gate_depth(suffix_transitions, left_class,
                                             unit_class) == 0u)
            ++commit_agreement_deficit;
        }
      }
      if (i + 1u < extent) {
        const std::uint32_t following =
            tokens[index * kConstructionMaxTokens + i + 1u];
        if (!token_is_slot(following)) {
          const std::uint32_t right_class = construction_suffix_class(
              unit_lengths, unit_content, unit_words, following, boundary_mask);
          if (right_class < kSuffixClassCount &&
              construction_suffix_gate_depth(suffix_transitions, unit_class,
                                             right_class) == 0u)
            ++commit_agreement_deficit;
        }
      }
    }
    const std::uint32_t commit_support_depth =
        resident_roles::integer_log_depth(supports[index]);
    unsigned long long score = (1ull + commit_support_depth) *
                               (1ull + slots + 8ull * role_matches) * 256ull;
    const std::uint32_t identity =
        construction_pattern_hash(tokens, index, extent);
    const std::uint32_t jitter = construction_mix(identity ^ resident_variety);
    score = (score * (192ull + (jitter & 63ull))) >> 8u;
    score >>= 4u * (commit_agreement_deficit > 15u ? 15u
                                                   : commit_agreement_deficit);
    if (last_selected[0] == identity) score >>= 2u;
    if (score == 0u) return;
    const std::uint32_t bounded_score =
        score > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(score);
    atomicMax(best_packed,
              (static_cast<unsigned long long>(bounded_score) << 32u) | identity);
    return;
  }

  // Slot-role demand must be covered by the active pool (multiset cover).
  std::uint32_t needed[kConstructionMaxSlots];
  std::uint32_t needed_counts[kConstructionMaxSlots];
  std::uint32_t distinct = 0u;
  for (std::uint32_t i = 0u; i < extent; ++i) {
    const std::uint32_t token = tokens[index * kConstructionMaxTokens + i];
    if (!token_is_slot(token)) continue;
    const std::uint32_t role = token_role(token);
    bool found = false;
    for (std::uint32_t d = 0u; d < distinct; ++d) {
      if (needed[d] == role) {
        ++needed_counts[d];
        found = true;
        break;
      }
    }
    if (!found && distinct < kConstructionMaxSlots) {
      needed[distinct] = role;
      needed_counts[distinct] = 1u;
      ++distinct;
    }
  }
  const std::uint32_t pool_count = pool_meta[0];
  if (pool_count == 0u) return;
  std::uint32_t diversity_deficit = 0u;
  for (std::uint32_t d = 0u; d < distinct; ++d) {
    if (needed[d] >= kConstructionRoleCount) return;
    // Binding deliberately permits reuse with a penalty, so one active filler
    // can lawfully cover repeated instances of the same structural role --
    // but frames whose demand exceeds the pool's diversity are demoted so a
    // frame that can be filled with DISTINCT words wins when one exists.
    const std::uint32_t available = pool_meta[1u + needed[d]];
    if (available == 0u) return;
    if (needed_counts[d] > available)
      diversity_deficit += needed_counts[d] - available;
  }

  unsigned long long activation = 0u;
  for (std::uint32_t k = 0u; k < pool_count; ++k) {
    for (std::uint32_t d = 0u; d < distinct; ++d) {
      if (pool_roles[k] == needed[d]) {
        activation += pool_weights[k];
        break;
      }
    }
  }

  // MORPHOLOGICAL AGREEMENT at frame choice (learned, lesionable): a frame
  // whose slot demands a role NO active candidate can fill with learned
  // suffix-class compatibility toward the frame's own literal neighbours is
  // demoted -- this is where "I <slot> have" loses when every candidate for
  // that slot is a "-tion" noun the stream never attested after "I ".
  // Slot-slot junctions are unknowable here and are left to the bind gate.
  std::uint32_t agreement_deficit = 0u;
  if (suffix_transitions != nullptr) {
    for (std::uint32_t i = 0u; i < extent; ++i) {
      const std::uint32_t token = tokens[index * kConstructionMaxTokens + i];
      if (!token_is_slot(token)) continue;
      const std::uint32_t role = token_role(token);
      std::uint32_t left_class = kNoSuffixClass;
      std::uint32_t right_class = kNoSuffixClass;
      if (i != 0u) {
        const std::uint32_t previous =
            tokens[index * kConstructionMaxTokens + i - 1u];
        if (!token_is_slot(previous))
          left_class = construction_suffix_class(
              unit_lengths, unit_content, unit_words, previous, boundary_mask);
      }
      if (i + 1u < extent) {
        const std::uint32_t following =
            tokens[index * kConstructionMaxTokens + i + 1u];
        if (!token_is_slot(following))
          right_class = construction_suffix_class(
              unit_lengths, unit_content, unit_words, following, boundary_mask);
      }
      if (left_class >= kSuffixClassCount && right_class >= kSuffixClassCount)
        continue;  // no literal junction to check for this slot
      // A candidate passes only when EVERY literal junction it would touch is
      // an attested suffix-class transition (summing junctions was tried and
      // let "I stagnation have" through on weak right-side evidence alone).
      bool fillable = false;
      for (std::uint32_t k = 0u; k < pool_count && !fillable; ++k) {
        if (pool_roles[k] != role) continue;
        const std::uint32_t candidate_class = construction_suffix_class(
            unit_lengths, unit_content, unit_words, pool_units[k],
            boundary_mask);
        const bool left_ok =
            left_class >= kSuffixClassCount ||
            construction_suffix_gate_depth(suffix_transitions, left_class,
                                           candidate_class) != 0u;
        const bool right_ok =
            right_class >= kSuffixClassCount ||
            construction_suffix_gate_depth(suffix_transitions, candidate_class,
                                           right_class) != 0u;
        fillable = left_ok && right_ok;
      }
      if (!fillable) ++agreement_deficit;
    }
  }
  const std::uint32_t support_depth =
      resident_roles::integer_log_depth(supports[index]);
  unsigned long long score =
      (1ull + support_depth) * (activation + 1ull) * (1ull + slot_counts[index]);
  // Resident variety: deterministic per-(skeleton, reply) jitter so different
  // replies exercise different learned frames; demote the immediately
  // previous winner.
  const std::uint32_t identity = construction_pattern_hash(tokens, index, extent);
  const std::uint32_t jitter = construction_mix(identity ^ resident_variety);
  score = (score * (192ull + (jitter & 63ull))) >> 8u;
  score >>= 3u * (diversity_deficit > 20u ? 20u : diversity_deficit);
  score >>= 4u * (agreement_deficit > 15u ? 15u : agreement_deficit);
  if (last_selected[0] == identity) score >>= 2u;
  if (score == 0u) return;
  const std::uint32_t bounded_score =
      score > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(score);
  const unsigned long long packed =
      (static_cast<unsigned long long>(bounded_score) << 32u) | identity;
  atomicMax(best_packed, packed);
}

// Attestation of a learned transition (previous -> next) in a sorted bigram
// table. This is the adult's own grammar matter; using it to score a filler
// against its FRAME NEIGHBOURS is local-grammar checking, not query-span
// retrieval (the cue bytes are nowhere in this path).
template <typename BigramKeyT>
[[nodiscard]] __device__ inline bool construction_bigram_attested(
    const BigramKeyT* bigrams, const std::uint32_t* counts,
    std::uint32_t bigram_count, std::uint32_t previous, std::uint32_t next) {
  std::uint32_t low = 0u;
  std::uint32_t high = bigram_count;
  while (low < high) {
    const std::uint32_t middle = (low + high) / 2u;
    const BigramKeyT key = bigrams[middle];
    if (key.previous < previous ||
        (key.previous == previous && key.next < next)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low < bigram_count && bigrams[low].previous == previous &&
         bigrams[low].next == next && counts[low] != 0u;
}

// ---------------------------------------------------------------------------
// BIND + REALIZE: fill each slot with the strongest role-matching active
// content unit (reuse penalised, transition-attested against its frame
// neighbours), literals pass through. Serial (tiny).
// ---------------------------------------------------------------------------
template <typename BigramKeyT>
static __global__ void bind_construction_kernel(
    const std::uint32_t* tokens, const std::uint32_t* lengths,
    const std::uint32_t* slot_counts, const std::uint32_t* construction_count,
    const unsigned long long* best_packed,
    const std::uint32_t* pool_units, const std::uint32_t* pool_roles,
    const std::uint32_t* pool_weights, const std::uint32_t* pool_meta,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* initial_form_mask,
    const BigramKeyT* bigrams, const std::uint32_t* bigram_counts,
    std::uint32_t bigram_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* suffix_transitions,
    const std::uint32_t* commitment_units, std::uint32_t* commitment_meta,
    std::uint32_t* last_selected, std::uint32_t* plan,
    std::uint32_t* plan_meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  plan_meta[0] = 0u;
  plan_meta[1] = kNoConstruction;
  plan_meta[2] = 0u;
  // Same resident variety seed as selection, and captured BEFORE this kernel
  // overwrites last_selected[0] with the new choice further down.
  const std::uint32_t resident_variety =
      construction_mix(last_selected[0] ^ (construction_count[0] * 0x9e3779b9u));
  const unsigned long long packed = best_packed[0];
  if (packed == 0ull) return;
  const std::uint32_t identity = static_cast<std::uint32_t>(packed);
  const std::uint32_t total =
      construction_count[0] < kConstructionCap ? construction_count[0] : kConstructionCap;
  std::uint32_t index = kNoConstruction;
  for (std::uint32_t candidate = 0u; candidate < total; ++candidate) {
    const std::uint32_t candidate_extent = lengths[candidate];
    if (candidate_extent != 0u &&
        construction_pattern_hash(tokens, candidate, candidate_extent) == identity) {
      index = candidate;
      break;
    }
  }
  if (index == kNoConstruction) return;
  const std::uint32_t extent = lengths[index];
  if (extent == 0u || extent > kConstructionMaxTokens) return;

  // CONTENT-COMMITMENT MODE: dumb serialization. The committed content units
  // are expressed IN ORDER -- each slot takes the next committed unit, the
  // frame's literals pass through as connective tissue, and the resident
  // serialization cursor advances on-device. The composer renders the
  // commitment; it does not re-decide content.
  if (commitment_units != nullptr && commitment_meta != nullptr &&
      commitment_meta[0] > commitment_meta[1]) {
    const std::uint32_t cursor = commitment_meta[1];
    std::uint32_t consumed = 0u;
    std::uint32_t serialized = 0u;
    for (std::uint32_t i = 0u; i < extent; ++i) {
      const std::uint32_t token = tokens[index * kConstructionMaxTokens + i];
      if (!token_is_slot(token)) {
        plan[serialized++] = token;
        continue;
      }
      if (cursor + consumed >= commitment_meta[0]) return;  // legacy path runs
      plan[serialized++] = commitment_units[cursor + consumed++];
    }
    plan_meta[0] = serialized;
    plan_meta[1] = index;
    plan_meta[2] = slot_counts[index];
    commitment_meta[1] = cursor + consumed;
    last_selected[0] = identity;
    return;
  }

  const std::uint32_t pool_count = pool_meta[0];
  if (pool_count == 0u) return;

  std::uint32_t used[kConstructionPoolCap];
  for (std::uint32_t k = 0u; k < pool_count; ++k) used[k] = 0u;

  std::uint32_t written = 0u;
  for (std::uint32_t i = 0u; i < extent; ++i) {
    const std::uint32_t token = tokens[index * kConstructionMaxTokens + i];
    if (!token_is_slot(token)) {
      plan[written++] = token;
      continue;
    }
    const std::uint32_t role = token_role(token);
    std::uint32_t chosen = kNoConstruction;
    unsigned long long chosen_score = 0u;
    // MORPHOLOGICAL AGREEMENT (learned, lesionable): per-candidate evidence
    // that this filler's byte-suffix class is compatible with its frame
    // neighbours -- the already-realized left unit and the known right
    // literal. Exact bigram attestation below is binary and sparse; this
    // generalizes it: every "-tion is" in the stream taught the
    // (-tion-class -> -is-class) cell, so "stagnation" prefers a
    // singular-agreeing continuation even when the exact word pair was never
    // seen. Candidates with zero evidence are barred from the strict passes
    // whenever some candidate has evidence (a real gate, with the anything-
    // goes pass 2 keeping frames fillable). nullptr (lesion) restores exact
    // legacy scoring.
    std::uint32_t agreement_depth[kConstructionPoolCap];
    bool agreement_pass[kConstructionPoolCap];
    bool any_agreement_pass = false;
    if (suffix_transitions != nullptr) {
      std::uint32_t right_class = kNoSuffixClass;
      if (i + 1u < extent) {
        const std::uint32_t following =
            tokens[index * kConstructionMaxTokens + i + 1u];
        if (!token_is_slot(following))
          right_class = construction_suffix_class(
              unit_lengths, unit_content, unit_words, following, boundary_mask);
      }
      const std::uint32_t left_class =
          written != 0u ? construction_suffix_class(unit_lengths, unit_content,
                                                    unit_words,
                                                    plan[written - 1u],
                                                    boundary_mask)
                        : kNoSuffixClass;
      for (std::uint32_t k = 0u; k < pool_count; ++k) {
        agreement_depth[k] = 0u;
        agreement_pass[k] = false;
        if (pool_roles[k] != role) continue;
        const std::uint32_t candidate_class = construction_suffix_class(
            unit_lengths, unit_content, unit_words, pool_units[k],
            boundary_mask);
        const std::uint32_t left_depth = construction_suffix_gate_depth(
            suffix_transitions, left_class, candidate_class);
        const std::uint32_t right_depth = construction_suffix_gate_depth(
            suffix_transitions, candidate_class, right_class);
        agreement_depth[k] = left_depth + right_depth;
        // A candidate agrees only when EVERY known junction is an attested
        // suffix-class transition (one-sided evidence let "stagnation have"
        // ride past a zero-evidence "I stagnation" junction).
        agreement_pass[k] =
            (left_class >= kSuffixClassCount || left_depth != 0u) &&
            (right_class >= kSuffixClassCount || right_depth != 0u);
        any_agreement_pass |= agreement_pass[k];
      }
    }
    // Three passes: (0) distinct fillers only, no sentence-initial forms in
    // non-initial slots; (1) allow reuse, still no initial forms mid-frame;
    // (2) anything of the right role. Pass 0 kills "talents talent talent,"
    // chains AND mid-sentence 'What'/'Why' operator forms.
    for (std::uint32_t pass = 0u; pass < 3u && chosen == kNoConstruction; ++pass) {
      for (std::uint32_t k = 0u; k < pool_count; ++k) {
        if (pool_roles[k] != role) continue;
        if (pass < 2u && i != 0u &&
            initial_form_mask[pool_units[k]] != 0u)
          continue;
        // Agreement gate: in the strict passes a candidate with an
        // unattested junction loses to any candidate whose every known
        // junction carries learned suffix-compatibility evidence.
        if (suffix_transitions != nullptr && pass < 2u && any_agreement_pass &&
            !agreement_pass[k])
          continue;
        if (pass == 0u) {
          bool near_duplicate = false;
          for (std::uint32_t j = 0u; j < pool_count && !near_duplicate; ++j) {
            if (used[j] == 0u || j == k) continue;
            const std::uint32_t span =
                min(4u, min(unit_lengths[pool_units[k]],
                            unit_lengths[pool_units[j]]));
            bool same = span != 0u;
            for (std::uint32_t b = 0u; b < span && same; ++b) {
              same = construction_unit_byte(unit_content, unit_words,
                                            pool_units[k], b) ==
                     construction_unit_byte(unit_content, unit_words,
                                            pool_units[j], b);
            }
            near_duplicate = same;
          }
          if (near_duplicate || used[k] != 0u) continue;
        }
        unsigned long long weight = pool_weights[k];
        // Reuse penalty keeps fillers distinct so the realized content
        // sequence never mirrors one resident sentence.
        weight >>= 4u * used[k] > 62u ? 62u : 4u * used[k];
        const std::uint32_t jitter =
            construction_mix(pool_units[k] ^ (resident_variety + i) * 0x85ebca6bu);
        unsigned long long score = (weight + 1ull) * (224ull + (jitter & 31ull));
        // Local-grammar attestation: strongly prefer fillers whose junction
        // with the previous realized word and the following literal is a
        // transition the adult has actually learned.
        if (written != 0u &&
            construction_bigram_attested(bigrams, bigram_counts, bigram_count,
                                         plan[written - 1u], pool_units[k]))
          score *= 8ull;
        if (i + 1u < extent) {
          const std::uint32_t following =
              tokens[index * kConstructionMaxTokens + i + 1u];
          if (!token_is_slot(following) &&
              construction_bigram_attested(bigrams, bigram_counts, bigram_count,
                                           pool_units[k], following))
            score *= 8ull;
        }
        // Learned suffix-compatibility preference (see gate above): stronger
        // agreement evidence with the frame neighbours wins within a role.
        if (suffix_transitions != nullptr) {
          const std::uint32_t depth = agreement_depth[k];
          score <<= depth > kSuffixGateMaxShift ? kSuffixGateMaxShift : depth;
        }
        if (score > chosen_score) {
          chosen_score = score;
          chosen = k;
        }
      }
    }
    if (chosen == kNoConstruction) {
      plan_meta[0] = 0u;
      return;  // unfillable after all: leave plan empty, legacy path runs
    }
    ++used[chosen];
    plan[written++] = pool_units[chosen];
  }
  plan_meta[0] = written;
  plan_meta[1] = index;
  plan_meta[2] = slot_counts[index];
  last_selected[0] = identity;
}

// ---------------------------------------------------------------------------
// REALIZE bytes: literal function words + bound content words in
// construction order (units carry their own separators).
// ---------------------------------------------------------------------------
static __global__ void emit_construction_plan_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* plan,
    const std::uint32_t* plan_meta, const std::uint32_t* boundary_mask,
    std::uint8_t* output, std::uint32_t output_bytes,
    std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t written = 0u;
  const std::uint32_t count = plan_meta[0];
  std::uint32_t last_byte = 0x20u;  // treat start-of-reply as post-space
  for (std::uint32_t item = 0u; item < count && written < output_bytes; ++item) {
    const std::uint32_t unit = plan[item];
    const std::uint32_t length = unit_lengths[unit];
    if (length == 0u) continue;
    const std::uint32_t first =
        unit_content[unit * unit_words] & 0xffu;
    // Insert exactly one separating space between adjacent WORD units so units
    // never fuse ("not"+"completely" -> "not completely"). Skip when a boundary
    // byte (space/punctuation) already sits on either side.
    if (item != 0u && written < output_bytes &&
        boundary_mask[last_byte] == 0u && boundary_mask[first] == 0u) {
      output[written++] = 0x20u;
      last_byte = 0x20u;
    }
    for (std::uint32_t offset = 0u; offset < length && written < output_bytes;
         ++offset) {
      const std::uint32_t word = unit_content[unit * unit_words + offset / 4u];
      const std::uint8_t byte =
          static_cast<std::uint8_t>(word >> ((offset % 4u) * 8u));
      output[written++] = byte;
      last_byte = byte;
    }
  }
  generated_count[0] = written;
}
