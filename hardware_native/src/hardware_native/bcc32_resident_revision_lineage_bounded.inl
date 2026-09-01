// ---------------------------------------------------------------------
// GitHub #1060 (duplicate #1162 closed onto this ticket): iterative,
// capacity-bounded replacement for the revision-transfer authority mutual
// recursion.
//
// The cycle (measured against current `main`; see
// docs/diary/2026-08-16/.../nvlink-cannot-bound-the-epoch-kernel...md) is:
//
//   resident_program_authoritative                 (A, anti_unification.cuh)
//     -> resident_program_shadowed_by_revision      (B, above, ~line 1332)
//     -> resident_revision_transfer_product_authoritative (C, ~line 1192)
//   C -> resident_revision_participation_reader_authoritative (D, ~line 1040)
//   D -> ticketed_revision_source_authoritative     (E, ~line 694)
//   D -> revision_source_return_lineage             (H, ~line 764) -> E
//   E -> ticketed_return_witness_authoritative      (F, ~line 607)
//   E -> revision_intervention_lineage_authoritative(G, ~line 27)
//   F -> G
//   G -> A                                          [closes the cycle]
//
// (line numbers refer to bcc32_resident_revision_transfer.inl, which
// includes this file and defines A/B/C/D/E/F/G/H)
//
// `748220e41d` threaded a defaulted `recursion_depth` parameter through A,
// B, C, D, E, F, G as a runtime safety net, checked against a conservative
// bound only at A's entry. Its own commit message says plainly that nvlink
// still reports `resident_rewrite_epoch_kernel` / `resident_epoch_post_
// return_kernel` as STACK:UNKNOWN with the guard in place: a runtime depth
// check does not change the static call-graph shape nvlink walks, because
// the cycle is still a cycle in the compiled object code no matter what
// runtime guard sits inside it. The only fix is to remove the back edge
// (G -> A) from the object code entirely. That in turn means nothing
// reachable from G's *replacement* may call back into A, B, C, D, E, F, G,
// or H either, or the same cycle reappears one level down (G's replacement
// calling unmodified E, which calls G, which calls G's replacement again).
// So every function on the path from A down to G is reimplemented below,
// calling only each other in one strict forward direction (F' first,
// nothing below depends on anything above it) plus the same genuinely-leaf
// predicates the originals called. A, B, C, D, E, F, and H THEMSELVES
// remain completely unchanged in bcc32_resident_revision_transfer.inl
// (aside from `748220e41d`'s unrelated recursion_depth threading) and stay
// reachable from every one of their other ~40 call sites across the
// codebase exactly as before; only G's body was replaced there, with a
// one-line forward into `revision_intervention_lineage_authoritative_
// bounded` below.
//
// The one genuine recursive fan-out in the whole chain is G's own
// "alternative program" check (originally line ~122 of the parent file):
// to decide if an inquiry's lineage is authoritative, G needs to know if
// each of two OTHER program records is itself authoritative -- and per A's
// dispatch, that program could itself be a revision-transfer product with
// its own donor inquiry, needing the same G-shaped check again. That is a
// real mutual dependency between "is this program authoritative" (a
// kProgram question) and "is this inquiry's lineage authoritative" (a
// kInquiry question), so it cannot be flattened into a single top-to-bottom
// pass. Instead it is solved as a small fixed-point relaxation: every
// predicate below is pure (read-only over `const ResidentRewriteState*`),
// so the order in which its internal checks run cannot change the final
// answer -- only whether a check is "not decided yet" needs to be threaded
// through honestly. Each `evaluate_*` function below therefore takes a
// `bool* ready` out-parameter: whenever it would have needed a definitive
// answer from a dependency that is not resolved yet, it sets `*ready =
// false` and returns immediately (never a guessed value), and the driver
// retries every still-pending item once more of its dependencies have
// resolved. This is the same short-circuit-preserving property the
// original recursive code had (a resolved `false` dependency still fails
// the whole check immediately; only "not yet known" is deferred).
namespace revision_lineage_bounded {

// A well-formed (acyclic) record graph never needs more than
// live_record_capacity(state) distinct (kind, key) authority questions
// answered to resolve one root query -- there are only that many records to
// ask about in the first place. kCapacity additionally bounds the compiled
// frame size of the worklist itself. A state with more live records than
// this capacity fails closed exactly as it would on a genuine cycle,
// instead of growing this function's stack frame without bound -- which is
// precisely the defect this whole fix removes. Real sessions never come
// close to this: the diary's measured lineage chains are 1-3 levels deep.
inline constexpr std::uint32_t kCapacity = 32u;

enum class Kind : std::uint32_t { kProgram = 0u, kInquiry = 1u };
enum class Status : std::uint8_t { kPending = 0u, kFalse = 1u, kTrue = 2u };

struct Worklist {
  Kind kind[kCapacity]{};
  std::uint32_t key_a[kCapacity]{};
  std::uint32_t key_b[kCapacity]{};
  Status status[kCapacity]{};
  std::uint32_t count = 0u;
};

// Finds or creates the worklist entry for (kind, a, b). Returns kInvalid
// only when the item is new and the worklist is already at kCapacity;
// every caller below treats that exactly like an unresolved cycle (fail
// closed, never grow the array).
BCC32_REWRITE_HD inline std::uint32_t intern(Worklist* list, Kind kind,
                                             std::uint32_t a,
                                             std::uint32_t b) {
  for (std::uint32_t i = 0u; i < list->count; ++i)
    if (list->kind[i] == kind && list->key_a[i] == a && list->key_b[i] == b)
      return i;
  if (list->count >= kCapacity) return kInvalid;
  const std::uint32_t index = list->count++;
  list->kind[index] = kind;
  list->key_a[index] = a;
  list->key_b[index] = b;
  list->status[index] = Status::kPending;
  return index;
}

// Tri-state dependency reads. `*ready == false` always means "not decided
// this round, the driver will retry a later round" -- callers must not act
// on the returned bool unless `*ready` is true. These two functions are the
// only replacement for every call this cycle used to make back into
// `resident_program_authoritative` / `revision_intervention_lineage_
// authoritative`.
BCC32_REWRITE_HD inline bool check_program(Worklist* list, std::uint32_t slot,
                                           bool* ready) {
  const std::uint32_t index = intern(list, Kind::kProgram, slot, 0u);
  if (index == kInvalid) { *ready = true; return false; }
  *ready = list->status[index] != Status::kPending;
  return list->status[index] == Status::kTrue;
}

BCC32_REWRITE_HD inline bool check_inquiry(Worklist* list,
                                           std::uint32_t inquiry_owner,
                                           std::uint32_t inquiry_revision,
                                           bool* ready) {
  const std::uint32_t index =
      intern(list, Kind::kInquiry, inquiry_owner, inquiry_revision);
  if (index == kInvalid) { *ready = true; return false; }
  *ready = list->status[index] != Status::kPending;
  return list->status[index] == Status::kTrue;
}

// --- F: ticketed_return_witness_authoritative, iterative-safe body -----
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_ticketed_return_witness(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t witness_slot, std::uint32_t source_slot, bool* ready) {
  *ready = true;
  const bool require_source = source_slot != kInvalid;
  if (state == nullptr || witness_slot >= live_record_capacity(state) ||
      (require_source &&
       (source_slot >= live_record_capacity(state) ||
        !pure_external_exact_program_authoritative(state, source_slot))))
    return false;
  const Record& witness = state->records[witness_slot];
  if (witness.matter_q8 == 0u ||
      witness.lane[0] != kFormTicketedReturnWitness ||
      witness.lane[1] == 0u || witness.lane[1] == kInvalid ||
      witness.lane[4] == 0u || witness.lane[5] == 0u ||
      witness.lane[5] == kInvalid || witness.lane[6] == 0u ||
      witness.lane[6] == kInvalid ||
      witness.lane[7] != kCausalGermlineExternal ||
      witness.reserved[0] == 0u || witness.reserved[1] != 0u ||
      witness.lane[1] != ticketed_return_witness_owner(
                              witness.lane[5], witness.lane[2]))
    return false;
  bool inquiry_ready = false;
  const bool inquiry_ok = check_inquiry(list, witness.lane[6],
                                        witness.reserved[0], &inquiry_ready);
  if (!inquiry_ready) { *ready = false; return false; }
  if (!inquiry_ok) return false;
  std::uint32_t action_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& issued = state->records[slot];
    if (issued.matter_q8 == 0u ||
        issued.lane[0] != kFormRevisionActionIssuance ||
        issued.lane[1] != witness.lane[5])
      continue;
    if (issued.lane[7] != kCausalGermlineExternal ||
        issued.lane[2] >= live_record_capacity(state) ||
        issued.lane[6] != witness.lane[6] ||
        issued.reserved[0] != witness.reserved[0] ||
        !revision_ambiguity_trace_authoritative(state, issued) ||
        !revision_egress_witness_authoritative(state, issued, true) ||
        issued.lane[1] != revision_action_owner(
                              issued.lane[2], issued.lane[3], issued.lane[4],
                              issued.lane[6], issued.reserved[0]) ||
        state->records[issued.lane[2]].matter_q8 == 0u ||
        state->records[issued.lane[2]].revision != issued.lane[4] ||
        state->records[issued.lane[2]].lane[1] != issued.lane[5])
      return false;
    ++action_count;
  }
  if (action_count != 1u) return false;
  const Record* source =
      require_source ? &state->records[source_slot] : nullptr;
  if (require_source &&
      (witness.lane[2] != source->lane[1] ||
       witness.lane[4] != source->lane[2]))
    return false;
  std::uint32_t term_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u ||
        term.lane[0] != kFormTicketedReturnWitnessTerm ||
        term.lane[1] != witness.lane[1])
      continue;
    if (term.lane[2] >= witness.lane[4] ||
        term.lane[4] != witness.lane[2] ||
        term.lane[5] != witness.lane[3] ||
        term.lane[6] != witness.lane[4] ||
        term.lane[7] != kCausalGermlineExternal || term.revision != 1u)
      return false;
    if (require_source) {
      std::uint32_t source_word = 0u;
      std::uint32_t meta = 0u;
      if (!program_term_at(state, source->lane[1], term.lane[2], &source_word,
                           &meta) ||
          meta != 0u || source_word != term.lane[3])
        return false;
    }
    ++term_count;
  }
  for (std::uint32_t index = 0u; index < witness.lane[4]; ++index) {
    std::uint32_t owned = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      owned += term.matter_q8 != 0u &&
               term.lane[0] == kFormTicketedReturnWitnessTerm &&
               term.lane[1] == witness.lane[1] && term.lane[2] == index;
    }
    if (owned != 1u) return false;
  }
  return term_count == witness.lane[4];
}

// --- E: ticketed_revision_source_authoritative, iterative-safe body ----
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_ticketed_revision_source(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t source_slot, bool* ready) {
  *ready = true;
  if (state == nullptr || source_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, source_slot))
    return false;
  const Record& source = state->records[source_slot];
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& returned = state->records[slot];
    if (returned.matter_q8 == 0u ||
        returned.lane[0] != kFormRevisionTransferReturn ||
        returned.lane[3] != source.lane[1])
      continue;
    if (returned.lane[1] == 0u || returned.lane[1] == kInvalid ||
        returned.lane[2] != source.revision ||
        returned.lane[4] != source.lane[5] ||
        returned.lane[5] != source.lane[2] ||
        returned.lane[6] == 0u || returned.lane[6] == kInvalid ||
        returned.lane[7] != kCausalGermlineExternal ||
        returned.lane[1] != revision_return_owner(
                                source.lane[1], source.revision,
                                returned.lane[6]) ||
        returned.reserved[0] != 0u || returned.reserved[1] != 0u)
      return false;
    std::uint32_t witness_count = 0u;
    std::uint32_t action_owner = kInvalid;
    for (std::uint32_t witness_slot = 0u; witness_slot < live_record_capacity(state);
         ++witness_slot) {
      const Record& witness = state->records[witness_slot];
      if (witness.matter_q8 == 0u ||
          witness.lane[0] != kFormTicketedReturnWitness ||
          witness.lane[1] != returned.lane[6])
        continue;
      bool witness_ready = false;
      const bool witness_ok = evaluate_ticketed_return_witness(
          state, list, witness_slot, source_slot, &witness_ready);
      if (!witness_ready) { *ready = false; return false; }
      if (!witness_ok) return false;
      action_owner = witness.lane[5];
      ++witness_count;
    }
    if (witness_count != 1u) return false;
    std::uint32_t action_count = 0u;
    for (std::uint32_t action_slot = 0u; action_slot < live_record_capacity(state);
         ++action_slot) {
      const Record& issued = state->records[action_slot];
      if (issued.matter_q8 == 0u ||
          issued.lane[0] != kFormRevisionActionIssuance ||
          issued.lane[1] != action_owner)
        continue;
      if (issued.lane[7] != kCausalGermlineExternal ||
          issued.lane[2] >= live_record_capacity(state) ||
          issued.lane[1] != revision_action_owner(
                                issued.lane[2], issued.lane[3],
                                issued.lane[4], issued.lane[6],
                                issued.reserved[0]) ||
          state->records[issued.lane[2]].matter_q8 == 0u ||
          state->records[issued.lane[2]].revision != issued.lane[4] ||
          state->records[issued.lane[2]].lane[1] != issued.lane[5] ||
          !revision_ambiguity_trace_authoritative(state, issued) ||
          !revision_egress_witness_authoritative(state, issued, true))
        return false;
      bool inquiry_ready = false;
      const bool inquiry_ok = check_inquiry(list, issued.lane[6],
                                            issued.reserved[0], &inquiry_ready);
      if (!inquiry_ready) { *ready = false; return false; }
      if (!inquiry_ok) return false;
      ++action_count;
    }
    if (action_count != 1u) return false;
    ++count;
  }
  return count == 1u;
}

// --- H: revision_source_return_lineage, iterative-safe body ------------
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_source_return_lineage(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t source_slot, std::uint32_t* p4_owner,
    std::uint32_t* trace_owner, bool* ready) {
  *ready = true;
  if (p4_owner == nullptr || trace_owner == nullptr) return false;
  bool source_ready = false;
  const bool source_ok =
      evaluate_ticketed_revision_source(state, list, source_slot, &source_ready);
  if (!source_ready) { *ready = false; return false; }
  if (!source_ok) return false;
  const Record& source = state->records[source_slot];
  std::uint32_t returned_count = 0u;
  *p4_owner = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& returned = state->records[slot];
    if (returned.matter_q8 != 0u &&
        returned.lane[0] == kFormRevisionTransferReturn &&
        returned.lane[3] == source.lane[1]) {
      *p4_owner = returned.lane[6];
      ++returned_count;
    }
  }
  std::uint32_t action_owner = kInvalid;
  std::uint32_t p4_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& p4 = state->records[slot];
    if (p4.matter_q8 != 0u && p4.lane[0] == kFormTicketedReturnWitness &&
        p4.lane[1] == *p4_owner) {
      action_owner = p4.lane[5];
      ++p4_count;
    }
  }
  std::uint32_t action_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& issued = state->records[slot];
    if (issued.matter_q8 == 0u ||
        issued.lane[0] != kFormRevisionActionIssuance ||
        issued.lane[1] != action_owner)
      continue;
    if (!revision_ambiguity_trace_authoritative(state, issued)) return false;
    *trace_owner = issued.reserved[1];
    ++action_count;
  }
  return returned_count == 1u && p4_count == 1u && action_count == 1u;
}

// --- D (+ C, a pure alias in the original): resident_revision_
// participation_reader_authoritative / resident_revision_transfer_product_
// authoritative, iterative-safe body. Callers below invoke this directly
// wherever the original called either name.
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_participation_reader(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t program_slot, bool* ready) {
  *ready = true;
  if (state == nullptr || program_slot >= live_record_capacity(state))
    return false;
  const Record& product = state->records[program_slot];
  const std::uint32_t flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagRevisionTransferProduct;
  if (product.matter_q8 == 0u ||
      product.lane[0] != kFormRevisionParticipationReader ||
      product.lane[7] != flags || product.lane[2] < 3u ||
      product.lane[3] < kProgramMatureSupport || product.lane[4] == 0u ||
      product.lane[4] > kMaximumProgramVariables)
    return false;

  std::uint32_t witness_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness ||
        witness.lane[1] != product.lane[1])
      continue;
    if (witness_slot != kInvalid) return false;
    witness_slot = slot;
  }
  if (witness_slot == kInvalid) return false;
  const Record& witness = state->records[witness_slot];
  std::uint32_t source_use_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& source_use = state->records[slot];
    if (source_use.matter_q8 == 0u ||
        source_use.lane[0] != kFormRevisionTransferSourceUse ||
        source_use.lane[1] != product.lane[1])
      continue;
    if (source_use_slot != kInvalid) return false;
    source_use_slot = slot;
  }
  if (source_use_slot == kInvalid) return false;
  const Record& source_use = state->records[source_use_slot];
  const std::uint32_t source_slot =
      causal_program_slot_by_owner(state, witness.lane[3]);
  if (source_slot == kInvalid || witness.lane[2] != product.lane[1] ||
      witness.lane[7] != kCausalGermlineExternal ||
      source_use.lane[2] != witness.lane[3] ||
      source_use.lane[3] != witness.lane[5] ||
      source_use.lane[7] != kCausalGermlineExternal ||
      state->records[source_slot].lane[5] != witness.lane[5] ||
      witness.reserved[0] == 0u || witness.reserved[1] < 2u ||
      witness.reserved[1] > product.lane[2] ||
      product.lane[2] != state->records[source_slot].lane[2])
    return false;
  bool source_ready = false;
  const bool source_ok =
      evaluate_ticketed_revision_source(state, list, source_slot, &source_ready);
  if (!source_ready) { *ready = false; return false; }
  if (!source_ok) return false;
  std::uint32_t p4_owner = kInvalid;
  std::uint32_t trace_owner = kInvalid;
  bool lineage_ready = false;
  const bool lineage_ok = evaluate_source_return_lineage(
      state, list, source_slot, &p4_owner, &trace_owner, &lineage_ready);
  if (!lineage_ready) { *ready = false; return false; }
  if (!lineage_ok) return false;

  std::uint32_t prior_count = 0u;
  std::uint32_t delta_count = 0u;
  std::uint32_t mismatch = kInvalid;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  for (std::uint32_t index = 0u; index < product.lane[2]; ++index) {
    std::uint32_t product_word = 0u;
    std::uint32_t product_meta = 0u;
    std::uint32_t source_word = 0u;
    std::uint32_t source_meta = 0u;
    if (!revision_participation_reader_term_at(
            state, product.lane[1], index, &product_word, &product_meta) ||
        !program_term_at(state, state->records[source_slot].lane[1], index,
                         &source_word, &source_meta) ||
        source_meta != 0u || product_word != source_word)
      return false;
    std::uint32_t expected_meta = 0u;
    std::uint32_t before_word = kInvalid;
    if (index < witness.reserved[1]) {
      std::uint32_t prior_slot = kInvalid;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& prior = state->records[slot];
        if (prior.matter_q8 == 0u ||
            prior.lane[0] != kFormRevisionTransferPriorTerm ||
            prior.lane[1] != product.lane[1] || prior.lane[2] != index)
          continue;
        if (prior_slot != kInvalid || prior.lane[5] != witness.reserved[0] ||
            prior.lane[6] != witness.reserved[1] ||
            prior.lane[7] != kCausalGermlineEnabled)
          return false;
        prior_slot = slot;
      }
      if (prior_slot == kInvalid) return false;
      const Record& prior = state->records[prior_slot];
      before_word = prior.lane[3];
      expected_meta = prior.lane[4];
      if (expected_meta == 0u) {
        if (prior.lane[3] != source_word) {
          if (index + 1u != witness.reserved[1] || mismatch != kInvalid ||
              !same_raw_channel(prior.lane[3], source_word))
            return false;
          mismatch = index;
        }
      } else {
        const std::uint32_t variable = expected_meta - 1u;
        if (variable >= product.lane[4]) return false;
        const std::uint32_t bit = 1u << variable;
        if ((bound & bit) != 0u && binding[variable] != source_word)
          return false;
        binding[variable] = source_word;
        bound |= bit;
      }
      ++prior_count;
    }
    std::uint32_t focal_deltas = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& delta = state->records[slot];
      if (delta.matter_q8 == 0u ||
          delta.lane[0] != kFormRevisionParticipationDelta ||
          delta.lane[1] != product.lane[1] || delta.lane[2] != index)
        continue;
      const bool extension = index >= witness.reserved[1];
      if (++focal_deltas != 1u || delta.lane[3] != before_word ||
          delta.lane[4] != (extension ? kInvalid : expected_meta) ||
          delta.lane[5] != source_word || delta.lane[6] != expected_meta ||
          delta.lane[7] != (extension ? 2u : 1u) ||
          delta.reserved[0] != p4_owner || delta.reserved[1] != trace_owner)
        return false;
    }
    const bool changed = index >= witness.reserved[1] ||
                         before_word != source_word;
    if (focal_deltas != (changed ? 1u : 0u)) return false;
    delta_count += focal_deltas;
    if (expected_meta > product.lane[4])
        return false;
    if (product_meta != expected_meta) return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& prior = state->records[slot];
    if (prior.matter_q8 != 0u &&
        prior.lane[0] == kFormRevisionTransferPriorTerm &&
        prior.lane[1] == product.lane[1] && prior.lane[2] >= witness.reserved[1])
      return false;
    const Record& delta = state->records[slot];
    if (delta.matter_q8 != 0u &&
        delta.lane[0] == kFormRevisionParticipationDelta &&
        delta.lane[1] == product.lane[1] && delta.lane[2] >= product.lane[2])
      return false;
  }
  return prior_count == witness.reserved[1] && mismatch == witness.lane[6] &&
         delta_count != 0u;
}

// --- B: resident_program_shadowed_by_revision, iterative-safe body -----
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_shadowed_by_revision(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t program_slot, bool* ready) {
  *ready = true;
  bool self_ready = false;
  const bool self_ok =
      evaluate_participation_reader(state, list, program_slot, &self_ready);
  if (!self_ready) { *ready = false; return false; }
  if (self_ok) return true;
  if (!revision_base_program_authoritative(state, program_slot)) return false;
  const std::uint32_t owner = state->records[program_slot].lane[1];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness ||
        witness.lane[2] != owner)
      continue;
    const std::uint32_t product_slot =
        causal_program_slot_by_owner(state, witness.lane[1]);
    if (product_slot == kInvalid) continue;
    bool product_ready = false;
    const bool product_ok =
        evaluate_participation_reader(state, list, product_slot, &product_ready);
    if (!product_ready) { *ready = false; return false; }
    if (product_ok) return true;
  }
  return false;
}

// --- A: resident_program_authoritative dispatch, iterative-safe body ---
// A itself (bcc32_resident_cross_context_anti_unification.cuh) is
// UNCHANGED and remains what every other call site in the codebase uses.
// This copy of its dispatch exists only to resolve the two "alternative
// program" dependencies G's inquiry check produces, via the worklist below
// instead of a direct call.
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_program(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t program_slot, bool* ready) {
  *ready = true;
  if (state == nullptr || program_slot >= live_record_capacity(state))
    return false;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u ||
      (program.lane[7] & kProgramFlagEnabled) == 0u)
    return false;
  if (program.lane[0] == kFormSpanProgram) {
    if ((program.lane[7] & kProgramFlagCausalGermlineProduct) != 0u)
      return state->causal_germline_validation_pending == 0u &&
             causal_germline_span_product_authoritative(state, program_slot);
    return program.lane[3] >= kSpanProgramMatureSupport;
  }
  if (program.lane[0] == kFormProgram &&
      (program.lane[7] & kProgramFlagPureExternalExact) != 0u)
    return pure_external_exact_program_authoritative(state, program_slot) &&
           !pure_external_program_consumed_by_revision(state, program_slot) &&
           !causal_exact_program_shadowed_by_version_space(
               state, program_slot);
  if (program.lane[0] == kFormProgram &&
      (program.lane[7] & kProgramFlagVersionSpace) != 0u)
    return (program.lane[7] & kProgramFlagEnabled) != 0u &&
           version_space_program_authoritative(state, program_slot) &&
           !causal_product_has_live_counterevidence(state, program_slot);
  if (program.lane[0] == kFormProgram &&
      (program.lane[7] & kProgramFlagCausalGermlineProduct) != 0u)
    return state->causal_germline_validation_pending == 0u &&
           causal_germline_product_authoritative(state, program_slot);
  if (program.lane[0] == kFormProgram &&
      (program.lane[7] & kProgramFlagRevisionTransferProduct) != 0u) {
    bool product_ready = false;
    const bool product_ok = evaluate_participation_reader(
        state, list, program_slot, &product_ready);
    *ready = product_ready;
    return product_ready && product_ok;
  }
  if (program.lane[0] != kFormProgram) return false;
  if ((program.lane[7] & kProgramFlagResidentEvidenceOnly) != 0u)
    return cross_context::cross_context_program_is_derived_exact(
        state, program_slot);
  if (program.lane[3] < kProgramMatureSupport) return false;
  bool shadow_ready = false;
  const bool shadowed =
      evaluate_shadowed_by_revision(state, list, program_slot, &shadow_ready);
  *ready = shadow_ready;
  return shadow_ready && !shadowed;
}

// --- G: revision_intervention_lineage_authoritative, iterative-safe body,
// byte-for-byte the same checks as the original definition in the parent
// file, with the one recursive call replaced by `check_program` (marked
// below).
BCC32_CAUSAL_GERMLINE_DISPATCH bool evaluate_inquiry(
    const ResidentRewriteState* state, Worklist* list,
    std::uint32_t inquiry_owner, std::uint32_t inquiry_revision,
    bool* ready) {
  *ready = true;
  if (state == nullptr || inquiry_owner == 0u || inquiry_owner == kInvalid ||
      inquiry_revision == 0u)
    return false;
  std::uint32_t inquiry_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kRevisionInquiryHeaderForm ||
        candidate.lane[1] != inquiry_owner)
      continue;
    if (inquiry_slot != kInvalid) return false;
    inquiry_slot = slot;
  }
  if (inquiry_slot == kInvalid) return false;
  const Record& inquiry = state->records[inquiry_slot];
  if (inquiry.revision != inquiry_revision ||
      (inquiry.lane[7] & kRevisionInquiryRequiredState) !=
          kRevisionInquiryRequiredState ||
      (inquiry.lane[7] & kRevisionInquiryCapturedState) != 0u ||
      inquiry.lane[2] == 0u || inquiry.lane[2] == kInvalid ||
      inquiry.lane[3] == 0u || inquiry.lane[4] != 2u ||
      inquiry.lane[5] == 0u || inquiry.lane[5] == kInvalid ||
      inquiry.lane[6] == 0u || inquiry.reserved[0] != 0u ||
      inquiry.reserved[1] >= live_record_capacity(state))
    return false;

  const Record& constructor = state->records[inquiry.reserved[1]];
  if (constructor.matter_q8 == 0u ||
      constructor.lane[0] != kRevisionInquiryConstructorForm ||
      constructor.lane[1] == 0u || constructor.lane[1] == kInvalid ||
      constructor.lane[2] == 0u || constructor.lane[2] > 32u ||
      constructor.lane[3] != 3u || constructor.lane[7] != 1u)
    return false;
  std::uint32_t constructor_terms = 0u;
  std::uint32_t constructor_term_indices = 0u;
  std::uint32_t constructor_witnesses = 0u;
  std::uint32_t prior_episode[3]{kInvalid, kInvalid, kInvalid};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == kRevisionInquiryConstructorTermForm &&
        record.lane[1] == constructor.lane[1]) {
      if (record.lane[2] >= constructor.lane[2] ||
          (constructor_term_indices & (1u << record.lane[2])) != 0u)
        return false;
      constructor_term_indices |= 1u << record.lane[2];
      ++constructor_terms;
      continue;
    }
    if (record.lane[0] != kRevisionInquiryConstructorWitnessForm ||
        record.lane[1] != constructor.lane[1])
      continue;
    if (constructor_witnesses == 3u || record.lane[2] == 0u ||
        record.lane[2] == kInvalid)
      return false;
    for (std::uint32_t prior = 0u; prior < constructor_witnesses; ++prior)
      if (prior_episode[prior] == record.lane[2]) return false;
    std::uint32_t episodes = 0u;
    for (std::uint32_t episode_slot = 0u; episode_slot < live_record_capacity(state);
         ++episode_slot) {
      const Record& episode = state->records[episode_slot];
      if (episode.matter_q8 != 0u &&
          episode.lane[0] == kRevisionInquiryHeaderForm &&
          episode.lane[1] == record.lane[2] &&
          episode.revision == record.lane[3])
        ++episodes;
    }
    if (episodes != 1u) return false;
    prior_episode[constructor_witnesses++] = record.lane[2];
  }
  if (constructor_terms != constructor.lane[2] ||
      constructor_witnesses != 3u)
    return false;

  std::uint32_t alternatives = 0u;
  std::uint32_t selected = 0u;
  std::uint32_t selected_word = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kRevisionInquiryAlternativeForm ||
        binding.lane[1] != inquiry_owner)
      continue;
    std::uint32_t programs = 0u;
    for (std::uint32_t program_slot = 0u; program_slot < live_record_capacity(state);
         ++program_slot) {
      const Record& program = state->records[program_slot];
      if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
          program.lane[1] != binding.lane[2])
        continue;
      if (program.revision != binding.lane[3]) return false;
      // Originally `!resident_program_authoritative(state, program_slot)`
      // (the sole back edge that closed the cycle). A resolved `false`
      // here still fails the whole inquiry immediately, exactly as the
      // original `return false` did; "not yet known" defers to a later
      // round instead of recursing.
      bool program_ready = false;
      const bool program_ok =
          check_program(list, program_slot, &program_ready);
      if (!program_ready) { *ready = false; return false; }
      if (!program_ok) return false;
      ++programs;
    }
    if (programs != 1u) return false;
    ++alternatives;
    if (binding.lane[2] == inquiry.lane[5] &&
        binding.lane[3] == inquiry.lane[6]) {
      ++selected;
      selected_word = binding.lane[4];
    }
  }
  if (alternatives != 2u || selected != 1u || selected_word == kInvalid)
    return false;

  std::uint32_t replies = 0u;
  std::uint32_t resumes = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u || witness.lane[1] != inquiry_owner)
      continue;
    if (witness.lane[0] == kRevisionInquiryReplyWitnessForm) {
      if (witness.lane[3] != inquiry.lane[5] ||
          witness.lane[4] != inquiry.lane[6] ||
          witness.lane[5] != selected_word || witness.reserved[0] == 0u ||
          witness.reserved[0] > 32u ||
          witness.lane[7] != kRevisionInquiryExternalWitness)
        return false;
      std::uint32_t terms = 0u;
      std::uint32_t term_indices = 0u;
      for (std::uint32_t term_slot = 0u; term_slot < live_record_capacity(state);
           ++term_slot) {
        const Record& term = state->records[term_slot];
        if (term.matter_q8 == 0u ||
            term.lane[0] != kRevisionInquiryReplyTermForm ||
            term.lane[1] != inquiry_owner)
          continue;
        if (term.lane[4] != witness.lane[2] ||
            term.lane[5] != witness.lane[6] ||
            term.lane[6] != witness.reserved[0] ||
            term.lane[7] != kRevisionInquiryExternalWitness ||
            term.lane[2] >= witness.reserved[0] ||
            (term_indices & (1u << term.lane[2])) != 0u)
          return false;
        term_indices |= 1u << term.lane[2];
        ++terms;
      }
      if (terms != witness.reserved[0]) return false;
      ++replies;
    } else if (witness.lane[0] == kRevisionInquiryResumeWitnessForm) {
      if (witness.lane[2] != inquiry.lane[5] ||
          witness.lane[3] != inquiry.lane[6] ||
          witness.lane[4] != selected_word)
        return false;
      ++resumes;
    }
  }
  return replies == 1u && resumes == 1u;
}

// --- Driver: resolves a rooted (kInquiry, inquiry_owner, inquiry_revision)
// query to a fixed point over the bounded worklist above. This is the only
// function the real `revision_intervention_lineage_authoritative` (in the
// parent file) calls; it never calls back into A, B, C, D, E, F, G, or H,
// so the whole object-code graph from here down is a strict DAG. Each
// round evaluates every still-pending item once; an item that becomes
// ready gets a final status, and any dependency it discovered along the
// way is already interned (as kPending) for a later round to pick up. The
// loop stops as soon as the root resolves, or after a round makes no
// progress at all (which can only happen on a genuine cycle in the record
// graph or on hitting kCapacity -- both fail closed, matching a
// corrupted/pathological record graph getting rejected instead of
// recursing until the stack faults).
BCC32_CAUSAL_GERMLINE_DISPATCH bool
revision_intervention_lineage_authoritative_bounded(
    const ResidentRewriteState* state, std::uint32_t inquiry_owner,
    std::uint32_t inquiry_revision) {
  if (state == nullptr) return false;
  Worklist list{};
  const std::uint32_t root =
      intern(&list, Kind::kInquiry, inquiry_owner, inquiry_revision);
  if (root == kInvalid) return false;
  const std::uint32_t capacity = live_record_capacity(state);
  const std::uint32_t bound = capacity < kCapacity ? capacity : kCapacity;
  for (std::uint32_t round = 0u; round < bound; ++round) {
    bool progress = false;
    for (std::uint32_t i = 0u; i < list.count; ++i) {
      if (list.status[i] != Status::kPending) continue;
      bool ready = false;
      bool value = false;
      if (list.kind[i] == Kind::kInquiry)
        value = evaluate_inquiry(state, &list, list.key_a[i], list.key_b[i],
                                 &ready);
      else
        value = evaluate_program(state, &list, list.key_a[i], &ready);
      if (ready) {
        list.status[i] = value ? Status::kTrue : Status::kFalse;
        progress = true;
      }
    }
    if (list.status[root] != Status::kPending)
      return list.status[root] == Status::kTrue;
    if (!progress) break;
  }
  return false;
}

}  // namespace revision_lineage_bounded
