BCC32_OPEN_INQUIRY_HD inline bool live_resident_continuation_alternative(
    const ResidentRewriteState* state, std::uint32_t slot,
    const Record& suspended, AlternativeSnapshot* snapshot,
    bool require_enabled = true) {
  if (snapshot != nullptr) *snapshot = AlternativeSnapshot{};
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[slot];
  const bool version_space =
      (program.lane[7] & kProgramFlagVersionSpace) != 0u;
  const bool external_exact =
      (program.lane[7] & kProgramFlagPureExternalExact) != 0u;
  const bool revision_transfer =
      (program.lane[7] & kProgramFlagRevisionTransferProduct) != 0u;
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      (!version_space && !external_exact && !revision_transfer) ||
      (version_space &&
       !version_space_program_grounded(state, slot, require_enabled)) ||
      (external_exact &&
       !pure_external_exact_program_authoritative(state, slot)) ||
      (revision_transfer &&
       !resident_revision_transfer_product_authoritative(state, slot)) ||
      // A VersionSpace product remains a resident alternative after valid
      // disambiguating counterevidence: the counterevidence records the
      // competing continuation that keeps the fork physically explicit.
      // Ordinary exact products remain suppressed once counterevidence has
      // withdrawn their executable authority.
      (!version_space && causal_product_has_live_counterevidence(state, slot)) ||
      suspended.lane[2] >= program.lane[2])
    return false;
  std::uint32_t word = 0u;
  std::uint32_t meta = 0u;
  if (version_space) {
    for (std::uint32_t index = 0u; index < suspended.lane[2]; ++index) {
      std::uint32_t observed = 0u;
      std::uint32_t expected = 0u;
      if (!trajectory_word_at(state, suspended.lane[1], index, &observed) ||
          !version_space_effective_program_term_at(state, program, index,
                                                   &expected, &meta) ||
          meta != 0u || observed != expected)
        return false;
    }
    if (!version_space_effective_program_term_at(
            state, program, suspended.lane[2], &word, &meta))
      return false;
  } else {
    bool unbound = false;
    std::uint32_t stored_word = 0u;
    if (!match_program_prefix(state, program, suspended, &word, &unbound) ||
        unbound ||
        !resident_program_term_at(state, slot, suspended.lane[2],
                                  &stored_word, &meta) ||
        stored_word != word)
      return false;
  }
  if (meta != 0u)
    return false;
  if (snapshot != nullptr) {
    snapshot->owner = program.lane[1];
    snapshot->revision = program.revision;
    snapshot->consequence = word;
    snapshot->slot = slot;
  }
  return true;
}

// Emission needs the resident extent before it can advance one word.  The
// historical digest reader deliberately scans and validates the whole suffix
// and rejects suffixes beyond kMaximumSurfaceWords; that is useful for the
// small bootstrap surface, but it turns every long reply word into another
// full page/term traversal.  Keep that exact preflight for the bounded path.
// For a longer resident suffix, use the page-native extent and revalidate the
// word actually emitted through alternative_continuation_word_at.  The next
// word therefore remains authority-checked at its point of publication while
// no semantic document-length cap or kRecordCapacity scratch array is added.
BCC32_OPEN_INQUIRY_HD inline bool alternative_continuation_extent_for_emission(
    const ResidentRewriteState* state, const AlternativeSnapshot& alternative,
    std::uint32_t suspended_length, std::uint32_t* length,
    std::uint32_t* digest) {
  if (length == nullptr || digest == nullptr || state == nullptr ||
      alternative.slot >= live_record_capacity(state))
    return false;
  *length = 0u;
  *digest = 0u;
  const Record& program = state->records[alternative.slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      suspended_length >= program.lane[2])
    return false;
  const std::uint32_t count = program.lane[2] - suspended_length;
  if (count <= kMaximumSurfaceWords)
    return alternative_continuation_digest(state, alternative,
                                           suspended_length, length, digest);
  std::uint32_t first_word = 0u;
  if (!alternative_continuation_word_at(state, alternative, suspended_length,
                                        0u, &first_word))
    return false;
  *length = count;
  return true;
}

// The old fork collector performs a complete record scan, including a nested
// scan for every candidate, every time a surface word is emitted.  Keep the
// candidate set in the resident emission cursor instead: one bounded stride
// advances per device call, while the final selected Program is revalidated at
// the point of publication.  The cursor stores only physical record slots;
// no answer, word, or semantic relation is cached here.
inline constexpr std::uint32_t kDistributedCompetitionCursorBits = 10u;
inline constexpr std::uint32_t kDistributedCompetitionCursorMask =
    (1u << kDistributedCompetitionCursorBits) - 1u;
inline constexpr std::uint32_t kDistributedCompetitionSlotBits = 11u;
inline constexpr std::uint32_t kDistributedCompetitionSlotMask =
    (1u << kDistributedCompetitionSlotBits) - 1u;
inline constexpr std::uint32_t kDistributedCompetitionFirstShift =
    kDistributedCompetitionCursorBits;
inline constexpr std::uint32_t kDistributedCompetitionSecondShift =
    kDistributedCompetitionFirstShift + kDistributedCompetitionSlotBits;
inline constexpr std::uint32_t kDistributedCompetitionTerminal =
    kDistributedCompetitionCursorMask;
inline constexpr std::uint32_t kDistributedCompetitionScanStride = 8u;

// The terminal cursor value is reserved only after the final physical slot
// has been scanned; the slot field stores physical_slot + 1.  Keep these
// layout assumptions explicit so a future record-capacity change cannot
// silently truncate a resident cursor or collide with its terminal marker.
static_assert(kRecordCapacity <= kDistributedCompetitionTerminal + 1u,
              "distributed cursor cannot address every resident record");
static_assert(kRecordCapacity <= kDistributedCompetitionSlotMask,
              "distributed cursor slot encoding is too narrow");
static_assert(kDistributedCompetitionSecondShift +
                      kDistributedCompetitionSlotBits <=
                  sizeof(std::uint32_t) * 8u,
              "distributed cursor exceeds its packed word");

BCC32_OPEN_INQUIRY_HD inline std::uint32_t distributed_competition_pack(
    std::uint32_t cursor, std::uint32_t first, std::uint32_t second) {
  return (cursor & kDistributedCompetitionCursorMask) |
         ((first & kDistributedCompetitionSlotMask)
          << kDistributedCompetitionFirstShift) |
         ((second & kDistributedCompetitionSlotMask)
          << kDistributedCompetitionSecondShift);
}

BCC32_OPEN_INQUIRY_HD inline bool distributed_competition_cursor_terminal(
    std::uint32_t packed) {
  return (packed & kDistributedCompetitionCursorMask) ==
         kDistributedCompetitionTerminal;
}

BCC32_OPEN_INQUIRY_HD inline bool distributed_competition_step(
    const ResidentRewriteState* state, const Record& suspended,
    std::uint32_t continuation_kind, std::uint32_t* cursor_state,
    AlternativeSnapshot* selected) {
  if (state == nullptr || cursor_state == nullptr || selected == nullptr ||
      (continuation_kind != kTermFirstAlternative &&
       continuation_kind != kTermSecondAlternative))
    return false;
  *selected = AlternativeSnapshot{};
  const std::uint32_t packed = *cursor_state;
  std::uint32_t cursor = packed & kDistributedCompetitionCursorMask;
  std::uint32_t first =
      (packed >> kDistributedCompetitionFirstShift) &
      kDistributedCompetitionSlotMask;
  std::uint32_t second =
      (packed >> kDistributedCompetitionSecondShift) &
      kDistributedCompetitionSlotMask;
  const std::uint32_t capacity = live_record_capacity(state);
  if (cursor == kDistributedCompetitionTerminal) {
    // A terminal cursor with exactly one encoded candidate is a completed
    // selection. Two candidates are the fail-closed ambiguity marker.
    if (first == 0u || second != 0u || first - 1u >= capacity)
      return false;
    return live_resident_continuation_alternative(
        state, first - 1u, suspended, selected);
  }
  if (cursor >= capacity) return false;

  AlternativeSnapshot left{};
  AlternativeSnapshot right{};
  if (first != 0u &&
      !live_resident_continuation_alternative(state, first - 1u, suspended,
                                              &left))
    return false;
  if (second != 0u &&
      !live_resident_continuation_alternative(state, second - 1u, suspended,
                                              &right))
    return false;

  const std::uint32_t end =
      cursor + kDistributedCompetitionScanStride < capacity
          ? cursor + kDistributedCompetitionScanStride
          : capacity;
  for (std::uint32_t slot = cursor; slot < end; ++slot) {
    AlternativeSnapshot candidate{};
    if (!live_resident_continuation_alternative(state, slot, suspended,
                                                &candidate))
      continue;

    // A candidate that is a proper extension of any live resident candidate
    // cannot be the terminal continuation.  This preserves the old collector
    // semantics without retaining a kRecordCapacity scratch array.
    bool downstream_extension = false;
    for (std::uint32_t other_slot = 0u; other_slot < capacity; ++other_slot) {
      if (other_slot == slot) continue;
      AlternativeSnapshot other{};
      if (!live_resident_continuation_alternative(state, other_slot, suspended,
                                                  &other))
        continue;
      if (alternative_continuation_proper_prefix(
              state, other, candidate, suspended.lane[2])) {
        downstream_extension = true;
        break;
      }
    }
    if (downstream_extension) continue;

    if (first != 0u &&
        alternative_continuations_equal(state, candidate, left,
                                        suspended.lane[2])) {
      if (candidate.slot < left.slot) first = candidate.slot + 1u;
      left = candidate.slot < left.slot ? candidate : left;
      continue;
    }
    if (second != 0u &&
        alternative_continuations_equal(state, candidate, right,
                                        suspended.lane[2])) {
      if (candidate.slot < right.slot) second = candidate.slot + 1u;
      right = candidate.slot < right.slot ? candidate : right;
      continue;
    }
    if (first != 0u && alternative_continuation_proper_prefix(
                           state, candidate, left, suspended.lane[2])) {
      first = candidate.slot + 1u;
      left = candidate;
      continue;
    }
    if (first != 0u && alternative_continuation_proper_prefix(
                           state, left, candidate, suspended.lane[2]))
      continue;
    if (second != 0u && alternative_continuation_proper_prefix(
                            state, candidate, right, suspended.lane[2])) {
      second = candidate.slot + 1u;
      right = candidate;
      continue;
    }
    if (second != 0u && alternative_continuation_proper_prefix(
                            state, right, candidate, suspended.lane[2]))
      continue;
    if (first == 0u) {
      first = candidate.slot + 1u;
      left = candidate;
    } else if (second == 0u) {
      second = candidate.slot + 1u;
      right = candidate;
    } else {
      // The candidate has no live shorter-prefix or duplicate explanation and
      // would create a third terminal consequence.  Preserve ambiguity in the
      // cursor so later epochs remain silent rather than rescanning forever.
      *cursor_state = distributed_competition_pack(
          kDistributedCompetitionTerminal, first, candidate.slot + 1u);
      return false;
    }
  }

  if (end != capacity) {
    *cursor_state = distributed_competition_pack(end, first, second);
    return false;
  }
  if (first == 0u || second == 0u) {
    // A one-candidate terminal must not look like a completed selection:
    // reserve the second field as the failure marker in that case.  The
    // selected state is the only terminal encoding with first != 0 and
    // second == 0.
    *cursor_state =
        first == 0u
            ? distributed_competition_pack(kDistributedCompetitionTerminal,
                                            0u, 0u)
            : distributed_competition_pack(kDistributedCompetitionTerminal,
                                            0u, first);
    return false;
  }
  std::uint32_t left_word = 0u;
  std::uint32_t right_word = 0u;
  if (!alternative_continuation_word_at(state, left, suspended.lane[2], 0u,
                                        &left_word) ||
      !alternative_continuation_word_at(state, right, suspended.lane[2], 0u,
                                         &right_word))
    return false;
  if (right_word < left_word ||
      (right_word == left_word && right.owner < left.owner)) {
    const std::uint32_t temporary_slot = first;
    first = second;
    second = temporary_slot;
    const AlternativeSnapshot temporary = left;
    left = right;
    right = temporary;
  }
  const std::uint32_t selected_slot =
      continuation_kind == kTermFirstAlternative ? first : second;
  *cursor_state = distributed_competition_pack(
      kDistributedCompetitionTerminal, selected_slot, 0u);
  *selected = continuation_kind == kTermFirstAlternative ? left : right;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool collect_exact_fork(
    const ResidentRewriteState* state, const Record& suspended,
    AlternativeSnapshot alternatives[2]) {
  if (state == nullptr || alternatives == nullptr) return false;
  alternatives[0] = AlternativeSnapshot{};
  alternatives[1] = AlternativeSnapshot{};
  // Validate each live candidate once. The previous collector called the
  // authority reader again for every candidate pair, which became a
  // pathological 1024x1024 resident scan once grounded VersionSpace
  // alternatives remained visible after disambiguating counterevidence.
  // More than two terminal outcomes are already ambiguous for this contract;
  // the bounded cache keeps the physical fork finite and fails closed on
  // unusually large candidate populations.
  constexpr std::uint32_t kMaximumForkCandidates = 32u;
  AlternativeSnapshot candidates[kMaximumForkCandidates]{};
  std::uint32_t candidate_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    AlternativeSnapshot candidate{};
    if (!live_resident_continuation_alternative(state, slot, suspended,
                                                &candidate))
      continue;
    bool duplicate_consequence = false;
    for (std::uint32_t index = 0u; index < candidate_count; ++index) {
      if (candidate.consequence != candidates[index].consequence ||
          !alternative_continuations_equal(state, candidate, candidates[index],
                                           suspended.lane[2]))
        continue;
      duplicate_consequence = true;
      // Equivalent live Programs name one outcome. Slot order only selects the
      // producer receipt for that already-identical continuation.
      if (candidate.slot < candidates[index].slot) candidates[index] = candidate;
      break;
    }
    if (duplicate_consequence) continue;
    if (candidate_count == kMaximumForkCandidates) return false;
    candidates[candidate_count++] = candidate;
  }
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < candidate_count; ++index) {
    const AlternativeSnapshot& candidate = candidates[index];
    bool downstream_extension = false;
    for (std::uint32_t other_index = 0u; other_index < candidate_count;
         ++other_index) {
      if (other_index == index) continue;
      if (alternative_continuation_proper_prefix(
              state, candidates[other_index], candidate, suspended.lane[2])) {
        downstream_extension = true;
        break;
      }
    }
    if (downstream_extension) continue;
    if (count == 2u) return false;
    alternatives[count++] = candidate;
  }
  if (count != 2u) return false;
  std::uint32_t first_word = 0u;
  std::uint32_t second_word = 0u;
  if (!alternative_continuation_word_at(state, alternatives[0], suspended.lane[2],
                                        0u, &first_word) ||
      !alternative_continuation_word_at(state, alternatives[1], suspended.lane[2],
                                        0u, &second_word))
    return false;
  if (second_word < first_word ||
      (second_word == first_word &&
       alternatives[1].owner < alternatives[0].owner)) {
    const AlternativeSnapshot temporary = alternatives[0];
    alternatives[0] = alternatives[1];
    alternatives[1] = temporary;
  }
  return alternatives[0].owner != alternatives[1].owner &&
         alternatives[0].revision != 0u && alternatives[1].revision != 0u;
}

// The resumable competition path deliberately fails closed when its bounded
// cursor cannot publish on the current call. The complete resident collector
// is the canonical authority already used by the older fork path; use it once
// to recover a uniquely declared physical alternative rather than leaving a
// valid inquiry permanently silent. This does not choose a word or semantic
// answer: the constructor term selects only which of the two physically
// collected continuations is admissible, and the selected Program is
// revalidated before its cursor becomes terminal.
BCC32_OPEN_INQUIRY_HD inline bool distributed_competition_fallback_exact(
    const ResidentRewriteState* state, const Record& suspended,
    std::uint32_t continuation_kind, std::uint32_t* cursor_state,
    AlternativeSnapshot* selected) {
  if (state == nullptr || cursor_state == nullptr || selected == nullptr ||
      (continuation_kind != kTermFirstAlternative &&
       continuation_kind != kTermSecondAlternative))
    return false;
  AlternativeSnapshot alternatives[2]{};
  if (!collect_exact_fork(state, suspended, alternatives)) return false;
  const std::uint32_t selected_index =
      continuation_kind == kTermFirstAlternative ? 0u : 1u;
  AlternativeSnapshot validated{};
  const AlternativeSnapshot& candidate = alternatives[selected_index];
  if (candidate.slot >= live_record_capacity(state) ||
      !live_resident_continuation_alternative(
          state, candidate.slot, suspended, &validated))
    return false;
  if (validated.slot != candidate.slot || validated.owner != candidate.owner ||
      validated.revision != candidate.revision ||
      validated.consequence != candidate.consequence)
    return false;
  *selected = validated;
  *cursor_state = distributed_competition_pack(
      kDistributedCompetitionTerminal, validated.slot + 1u, 0u);
  return true;
}
