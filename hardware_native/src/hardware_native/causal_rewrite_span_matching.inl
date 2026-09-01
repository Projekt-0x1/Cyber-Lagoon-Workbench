BCC32_REWRITE_HD inline bool indexed_trajectory_word_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word) {
  return trajectory_word_at(state, owner, index, word);
}

inline constexpr std::uint32_t kSpanPrefixSearchBudget = 16384u;

struct SpanPrefixSearchResult {
  std::uint32_t count = 0u;
  std::uint32_t next_term = kInvalid;
  std::uint32_t next_offset = 0u;
  bool next_unbound = false;
  bool complete = false;
  bool exhausted = false;
  std::uint32_t (*bindings)[kMaximumVariableSpanEvents] = nullptr;
  std::uint32_t* lengths = nullptr;
};

BCC32_REWRITE_HD inline void record_span_prefix_solution(
    std::uint32_t next_term, std::uint32_t next_offset, bool next_unbound,
    bool complete,
    const std::uint32_t bindings[kMaximumProgramVariables]
                                [kMaximumVariableSpanEvents],
    const std::uint32_t lengths[kMaximumProgramVariables],
    SpanPrefixSearchResult* result) {
  if (result == nullptr || result->count > 1u) return;
  bool same = result->count != 0u && result->next_term == next_term &&
              result->next_offset == next_offset &&
              result->next_unbound == next_unbound &&
              result->complete == complete;
  if (same) {
    for (std::uint32_t variable = 0u;
         variable < kMaximumProgramVariables && same; ++variable) {
      if (result->lengths[variable] != lengths[variable]) {
        same = false;
        break;
      }
      for (std::uint32_t offset = 0u; offset < lengths[variable]; ++offset)
        if (result->bindings[variable][offset] !=
            bindings[variable][offset]) {
          same = false;
          break;
        }
    }
  }
  if (same) return;
  if (result->count != 0u) {
    result->count = 2u;
    return;
  }
  result->count = 1u;
  result->next_term = next_term;
  result->next_offset = next_offset;
  result->next_unbound = next_unbound;
  result->complete = complete;
  for (std::uint32_t variable = 0u; variable < kMaximumProgramVariables;
       ++variable) {
    result->lengths[variable] = lengths[variable];
    for (std::uint32_t offset = 0u; offset < lengths[variable]; ++offset)
      result->bindings[variable][offset] = bindings[variable][offset];
  }
}

// Search only at the first occurrence of an unbound variable. Literal runs
// and already-bound recurrences are consumed iteratively, so recursion depth
// is bounded by kMaximumProgramVariables rather than by the term count. The
// old matcher rejected a variable as soon as its next literal anchor occurred
// more than once. That made a later recurrence unable to resolve the earlier
// local ambiguity. This search keeps those bounded alternatives alive until
// the complete observed prefix either selects one physical binding or remains
// genuinely ambiguous.
BCC32_REWRITE_HD inline void search_span_prefix_bindings(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory,
    const std::uint32_t program_term_slots[kMaximumSpanProgramTerms],
    std::uint32_t term_index, std::uint32_t cursor,
    std::uint32_t bindings[kMaximumProgramVariables]
                          [kMaximumVariableSpanEvents],
    std::uint32_t lengths[kMaximumProgramVariables],
    std::uint32_t* examined, SpanPrefixSearchResult* result) {
  if (result == nullptr || result->count > 1u || result->exhausted) return;
  std::uint32_t frame_term[kMaximumProgramVariables]{};
  std::uint32_t frame_cursor[kMaximumProgramVariables]{};
  std::uint32_t frame_variable[kMaximumProgramVariables]{};
  std::uint32_t frame_channel[kMaximumProgramVariables]{};
  std::uint32_t frame_next_length[kMaximumProgramVariables]{};
  std::uint32_t frame_maximum_length[kMaximumProgramVariables]{};
  std::uint32_t depth = 0u;
  bool evaluate = true;

  while (true) {
    if (evaluate) {
      bool failed = false;
      bool branched = false;
      while (term_index < program.lane[2]) {
        std::uint32_t kind = 0u;
        std::uint32_t value = 0u;
        std::uint32_t channel = 0u;
        std::uint32_t reserved = 0u;
        if (!indexed_span_program_term_at(
                state, program_term_slots, term_index, &kind, &value,
                &channel, &reserved)) {
          failed = true;
          break;
        }
        if (kind == kSpanTermLiteral) {
          if (cursor == trajectory.lane[2]) {
            record_span_prefix_solution(term_index, 0u, false, false,
                                        bindings, lengths, result);
            failed = true;
            break;
          }
          std::uint32_t observed = 0u;
          if (!indexed_trajectory_word_at(
                  state, trajectory.lane[1], cursor, &observed) ||
              observed != value) {
            failed = true;
            break;
          }
          ++term_index;
          ++cursor;
          continue;
        }
        if (kind != kSpanTermVariable || value >= kMaximumProgramVariables) {
          failed = true;
          break;
        }
        const std::uint32_t variable = value;
        if (lengths[variable] != 0u) {
          if ((bindings[variable][0] & kRawChannelMask) != channel) {
            failed = true;
            break;
          }
          for (std::uint32_t offset = 0u; offset < lengths[variable];
               ++offset) {
            if (cursor == trajectory.lane[2]) {
              record_span_prefix_solution(term_index, offset, false, false,
                                          bindings, lengths, result);
              failed = true;
              break;
            }
            std::uint32_t observed = 0u;
            if (!indexed_trajectory_word_at(
                    state, trajectory.lane[1], cursor, &observed) ||
                observed != bindings[variable][offset]) {
              failed = true;
              break;
            }
            ++cursor;
          }
          if (failed) break;
          ++term_index;
          continue;
        }
        if (cursor == trajectory.lane[2]) {
          record_span_prefix_solution(term_index, 0u, true, false, bindings,
                                      lengths, result);
          failed = true;
          break;
        }
        if (depth >= kMaximumProgramVariables) {
          result->exhausted = true;
          return;
        }
        frame_term[depth] = term_index;
        frame_cursor[depth] = cursor;
        frame_variable[depth] = variable;
        frame_channel[depth] = channel;
        frame_next_length[depth] = 1u;
        frame_maximum_length[depth] =
            trajectory.lane[2] - cursor < kMaximumVariableSpanEvents
                ? trajectory.lane[2] - cursor
                : kMaximumVariableSpanEvents;
        branched = true;
        evaluate = false;
        break;
      }
      if (!failed && !branched) {
        if (cursor == trajectory.lane[2])
          record_span_prefix_solution(kInvalid, 0u, false, true, bindings,
                                      lengths, result);
        failed = true;
      }
      if (result->count > 1u || result->exhausted) return;
      if (failed) {
        if (depth == 0u) return;
        --depth;
        lengths[frame_variable[depth]] = 0u;
        evaluate = false;
      }
    }

    bool selected = false;
    while (frame_next_length[depth] <= frame_maximum_length[depth]) {
      const std::uint32_t span_length = frame_next_length[depth]++;
      if (++*examined > kSpanPrefixSearchBudget) {
        result->exhausted = true;
        return;
      }
      const std::uint32_t variable = frame_variable[depth];
      bool channel_ok = true;
      for (std::uint32_t offset = 0u; offset < span_length; ++offset) {
        std::uint32_t observed = 0u;
        if (!indexed_trajectory_word_at(
                state, trajectory.lane[1], frame_cursor[depth] + offset,
                &observed) ||
            (observed & kRawChannelMask) != frame_channel[depth]) {
          channel_ok = false;
          break;
        }
        bindings[variable][offset] = observed;
      }
      if (!channel_ok) continue;

      std::uint32_t anchor_values[kMaximumSpanProgramTerms]{};
      std::uint32_t anchor_length = 0u;
      for (std::uint32_t following = frame_term[depth] + 1u;
           following < program.lane[2]; ++following) {
        std::uint32_t following_kind = 0u;
        std::uint32_t following_value = 0u;
        std::uint32_t ignored_left = 0u;
        std::uint32_t ignored_right = 0u;
        if (!indexed_span_program_term_at(
                state, program_term_slots, following, &following_kind,
                &following_value, &ignored_left, &ignored_right)) {
          result->exhausted = true;
          return;
        }
        if (following_kind != kSpanTermLiteral) break;
        anchor_values[anchor_length++] = following_value;
      }
      const std::uint32_t anchor_position =
          frame_cursor[depth] + span_length;
      if (anchor_length != 0u) {
        const std::uint32_t available =
            trajectory.lane[2] - anchor_position;
        const std::uint32_t compared =
            available < anchor_length ? available : anchor_length;
        // One exact event is enough only when it is the complete observed
        // suffix. This lets a physical PAUSE immediately after punctuation
        // bind the preceding recurrent span without weakening interior anchor
        // selection, which still requires at least two observed events.
        if (compared < anchor_length && compared < 2u &&
            !(compared == 1u && anchor_position + compared ==
                                     trajectory.lane[2]))
          continue;
        bool anchor_matches = true;
        for (std::uint32_t offset = 0u; offset < compared; ++offset) {
          std::uint32_t observed = 0u;
          if (!indexed_trajectory_word_at(
                  state, trajectory.lane[1], anchor_position + offset,
                  &observed) ||
              observed != anchor_values[offset] ||
              !same_raw_channel(observed, anchor_values[offset])) {
            anchor_matches = false;
            break;
          }
        }
        if (!anchor_matches) continue;
      }
      lengths[variable] = span_length;
      term_index = frame_term[depth] + 1u;
      cursor = anchor_position;
      ++depth;
      evaluate = true;
      selected = true;
      break;
    }
    if (selected) continue;
    if (depth == 0u) return;
    --depth;
    lengths[frame_variable[depth]] = 0u;
    evaluate = false;
  }
}

// GitHub #1060 (device-stack frame inventory, 2026-08-17). This search owns
// ~11 KiB of automatic storage: program_term_slots[256] (1,024 B), the
// search_bindings[32][64] matrix (8,192 B), search_lengths[32] (128 B), and
// the frame_* worklist arrays that search_span_prefix_bindings inlines into
// it (768 B). Its callers each own the matching bindings[32][64]/lengths[32]
// pair (8,320 B), so a caller that inlines this whole block carries ~19.4 KiB.
//
// Left `inline`, nvcc folded that block into two callers that sit at
// *different depths of the same* resident_rewrite_epoch_kernel call chain, so
// nvlink summed one scratch buffer twice. That held before 4e0d6e2b00
// (run_pending_resume_device 19,768 + pending_means::select_causal_action
// 19,560) and still holds after it (cross_contact::consume_cross_contact_event
// 21,672 + cross_contact::compact_dormant_history 31,896).
//
// A __noinline__ boundary charges it once, as a max over the callees, and is
// a pure storage-placement change: the body, its capacities, its fail-closed
// returns and its ambiguity/budget semantics are untouched. Measured by fresh
// relinks of the registered mixed-provenance production contract:
//   resident_rewrite_epoch_kernel 103,912 -> 81,720 (pre-4e0d6e2b00 tree)
//   resident_rewrite_epoch_kernel  69,080 -> 47,184 (at 79b934dee0)
// Both arms ran on the real GPU under tools/gpu_runtime_lock.sh with
// byte-identical contract output and no runtime regression (23,500 ms ->
// 23,310 ms). Do not restore plain `inline` here without re-reading
// `cuobjdump -res-usage` on a relinked binary -- the cost is invisible in
// this file, and only nvlink's link-time bound can see it.
#if defined(__CUDACC__)
#define BCC32_SPAN_MATCH_PREFIX_DISPATCH \
  [[maybe_unused]] static __host__ __device__ __noinline__
#else
#define BCC32_SPAN_MATCH_PREFIX_DISPATCH [[maybe_unused]] inline
#endif
BCC32_SPAN_MATCH_PREFIX_DISPATCH std::uint32_t span_match_prefix_at(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t start_offset,
    std::uint32_t* next_term,
    std::uint32_t* next_offset, bool* next_unbound, bool* complete,
    bool* ambiguous,
    std::uint32_t bindings[kMaximumProgramVariables]
                    [kMaximumVariableSpanEvents],
    std::uint32_t lengths[kMaximumProgramVariables]) {
  if (start_offset > trajectory.lane[2]) return kSpanMatchNone;
  *next_term = kInvalid;
  *next_offset = 0u;
  *next_unbound = false;
  *complete = false;
  *ambiguous = false;
  std::uint32_t program_term_slots[kMaximumSpanProgramTerms];
  for (std::uint32_t index = 0u; index < kMaximumSpanProgramTerms; ++index)
    program_term_slots[index] = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == kFormSpanProgramTerm &&
        record.lane[1] == program.lane[1] &&
        record.lane[2] < kMaximumSpanProgramTerms)
      program_term_slots[record.lane[2]] = slot;
  }
  std::uint32_t search_bindings[kMaximumProgramVariables]
                               [kMaximumVariableSpanEvents]{};
  std::uint32_t search_lengths[kMaximumProgramVariables]{};
  SpanPrefixSearchResult result{};
  result.bindings = bindings;
  result.lengths = lengths;
  std::uint32_t examined = 0u;
  search_span_prefix_bindings(
      state, program, trajectory, program_term_slots,
      0u, start_offset, search_bindings, search_lengths, &examined, &result);
  if (result.exhausted || result.count > 1u) {
    *ambiguous = true;
    return kSpanMatchAmbiguous;
  }
  if (result.count == 0u) return kSpanMatchNone;
  *next_term = result.next_term;
  *next_offset = result.next_offset;
  *next_unbound = result.next_unbound;
  *complete = result.complete;
  return kSpanMatchPrefix;
}
