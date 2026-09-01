#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

namespace substrate::bcc32::causal_rewrite::cross_context {

#if defined(__CUDACC__)
#define BCC32_CROSS_CONTEXT_HD __host__ __device__
#define BCC32_CROSS_CONTEXT_DISPATCH static __host__ __device__ __noinline__
#else
#define BCC32_CROSS_CONTEXT_HD
#define BCC32_CROSS_CONTEXT_DISPATCH inline
#endif

// Cross-context factoring is deliberately narrower than ordinary Program
// execution. Long resident Programs remain executable, but they are not
// candidates for this bounded structural compiler. Keeping this independent
// of kMaximumTrajectoryEvents makes the CUDA stack cost explicit.
inline constexpr std::uint32_t kMaximumCrossContextTerms = 64u;
inline constexpr std::uint32_t kMaximumCrossContextContributors = 64u;
inline constexpr std::uint32_t kMaximumCrossContextDepth = 8u;
inline constexpr std::uint32_t kMaximumCrossContextLineage =
    1u + kMaximumCrossContextContributors * kMaximumCrossContextDepth;

struct CrossContextProgramPattern {
  std::uint32_t term_count = 0u;
  std::uint32_t variable_count = 0u;
  std::uint32_t digest = 0u;  // prefilter only; equality compares every term
  std::uint32_t term_meta[kMaximumCrossContextTerms]{};
  std::uint32_t term_value[kMaximumCrossContextTerms]{};
};

struct CrossContextPairTable {
  std::uint32_t left[kMaximumProgramVariables]{};
  std::uint32_t right[kMaximumProgramVariables]{};
  std::uint32_t count = 0u;
};

// Word count for a per-Record eligibility bitmask covering `population`
// live Records (see cross_context_factor_all_mature_programs below). Kept
// as a function of the live population rather than a kRecordCapacity-sized
// compile-time constant: population grows past one page via
// grow_resident_pages, and a page-0-sized bitmask would silently stop
// covering every donor beyond the first kRecordsPerPage Records.
BCC32_CROSS_CONTEXT_HD inline std::uint32_t cross_context_eligibility_words(
    std::uint32_t population) {
  return (population + 31u) / 32u;
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t cross_context_pattern_digest(
    const CrossContextProgramPattern& pattern) {
  std::uint32_t digest = rewrite_mix(0x8e3d2b19u, pattern.term_count,
                                     pattern.variable_count);
  for (std::uint32_t i = 0u; i < pattern.term_count; ++i)
    digest = rewrite_mix(digest, pattern.term_meta[i], pattern.term_value[i]);
  return digest == 0u || digest == kInvalid ? digest ^ 0x51f2a7c3u : digest;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_pattern_equal(
    const CrossContextProgramPattern& left,
    const CrossContextProgramPattern& right) {
  if (left.term_count != right.term_count ||
      left.variable_count != right.variable_count ||
      left.digest != right.digest)
    return false;
  for (std::uint32_t i = 0u; i < left.term_count; ++i)
    if (left.term_meta[i] != right.term_meta[i] ||
        left.term_value[i] != right.term_value[i])
      return false;
  return true;
}

// Extract and canonically number variables in term/Record order.  This makes
// the representation independent of allocation slots and donor traversal.
BCC32_CROSS_CONTEXT_HD inline bool cross_context_pattern_from_program(
    const ResidentRewriteState* state, const Record& program,
    CrossContextProgramPattern* out) {
  if (state == nullptr || out == nullptr || program.lane[0] != kFormProgram ||
      (program.lane[7] & kProgramFlagVersionSpace) != 0u ||
      program.lane[2] == 0u || program.lane[2] > kMaximumCrossContextTerms)
    return false;
  *out = CrossContextProgramPattern{};
  out->term_count = program.lane[2];
  std::uint32_t remap[kMaximumProgramVariables]{};
  for (std::uint32_t i = 0u; i < kMaximumProgramVariables; ++i)
    remap[i] = kInvalid;
  std::uint32_t next_variable = 0u;
  for (std::uint32_t i = 0u; i < out->term_count; ++i) {
    if (!program_term_at(state, program.lane[1], i, &out->term_value[i],
                         &out->term_meta[i]))
      return false;
    if (out->term_meta[i] == 0u) continue;
    const std::uint32_t old_variable = out->term_meta[i] - 1u;
    if (old_variable >= kMaximumProgramVariables) return false;
    if (remap[old_variable] == kInvalid) {
      if (next_variable >= kMaximumProgramVariables) return false;
      remap[old_variable] = next_variable++;
    }
    out->term_meta[i] = remap[old_variable] + 1u;
    out->term_value[i] &= kRawChannelMask;
  }
  if (program.lane[4] < next_variable) return false;
  out->variable_count = next_variable;
  out->digest = cross_context_pattern_digest(*out);
  return true;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_pair_at(
    const CrossContextPairTable& pairs, std::uint32_t left,
    std::uint32_t right, std::uint32_t* index) {
  for (std::uint32_t i = 0u; i < pairs.count; ++i)
    if (pairs.left[i] == left && pairs.right[i] == right) {
      if (index != nullptr) *index = i;
      return true;
    }
  return false;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_add_bidirectional_pair(
    CrossContextPairTable* pairs, std::uint32_t left, std::uint32_t right,
    std::uint32_t* index) {
  if (cross_context_pair_at(*pairs, left, right, index)) return true;
  for (std::uint32_t i = 0u; i < pairs->count; ++i)
    if (pairs->left[i] == left || pairs->right[i] == right) return false;
  if (pairs->count >= kMaximumProgramVariables) return false;
  const std::uint32_t fresh = pairs->count++;
  pairs->left[fresh] = left;
  pairs->right[fresh] = right;
  if (index != nullptr) *index = fresh;
  return true;
}

// Derive a pattern while preserving variable-vs-literal kind, both directions
// of variable equality, raw channel, literal anchors, and repeated roles.
BCC32_CROSS_CONTEXT_DISPATCH bool cross_context_derive_pattern(
    const ResidentRewriteState* state, const Record& left,
    const Record& right, CrossContextProgramPattern* out) {
  if (state == nullptr || out == nullptr || left.lane[0] != kFormProgram ||
      right.lane[0] != kFormProgram ||
      (left.lane[7] & kProgramFlagVersionSpace) != 0u ||
      (right.lane[7] & kProgramFlagVersionSpace) != 0u ||
      left.lane[2] == 0u ||
      left.lane[2] != right.lane[2] ||
      left.lane[2] > kMaximumCrossContextTerms)
    return false;
  CrossContextProgramPattern pattern{};
  pattern.term_count = left.lane[2];
  CrossContextPairTable donor_variables{};
  CrossContextPairTable changing_literals{};
  std::uint32_t literal_occurrences[kMaximumProgramVariables]{};
  std::uint32_t donor_occurrences[kMaximumProgramVariables]{};
  std::uint32_t variable_kind[kMaximumCrossContextTerms]{};
  std::uint32_t variable_index[kMaximumCrossContextTerms]{};
  std::uint32_t anchor_count = 0u;
  for (std::uint32_t i = 0u; i < pattern.term_count; ++i) {
    std::uint32_t left_value = 0u, left_meta = 0u;
    std::uint32_t right_value = 0u, right_meta = 0u;
    if (!program_term_at(state, left.lane[1], i, &left_value, &left_meta) ||
        !program_term_at(state, right.lane[1], i, &right_value, &right_meta))
      return false;
    const bool left_variable = left_meta != 0u;
    const bool right_variable = right_meta != 0u;
    if (left_variable != right_variable) return false;
    if (!left_variable) {
      if (left_value == right_value) {
        ++anchor_count;
        pattern.term_value[i] = left_value;
      } else {
        if (!same_raw_channel(left_value, right_value)) return false;
        std::uint32_t variable = 0u;
        if (!cross_context_add_bidirectional_pair(
                &changing_literals, left_value, right_value, &variable))
          return false;
        variable_kind[i] = 1u;
        variable_index[i] = variable;
        ++literal_occurrences[variable];
      }
    } else {
      const std::uint32_t left_variable = left_meta - 1u;
      const std::uint32_t right_variable = right_meta - 1u;
      if (left_variable >= kMaximumProgramVariables ||
          right_variable >= kMaximumProgramVariables ||
          !same_raw_channel(left_value, right_value))
        return false;
      std::uint32_t variable = 0u;
      if (!cross_context_add_bidirectional_pair(
              &donor_variables, left_variable, right_variable, &variable))
        return false;
      variable_kind[i] = 2u;
      variable_index[i] = variable;
      ++donor_occurrences[variable];
    }
  }
  if (anchor_count < 2u ||
      changing_literals.count + donor_variables.count >
          kMaximumProgramVariables)
    return false;
  pattern.variable_count = changing_literals.count + donor_variables.count;
  std::uint32_t repeated_positions = 0u, repeated_classes = 0u;
  for (std::uint32_t i = 0u; i < changing_literals.count; ++i) {
    if (literal_occurrences[i] >= 2u) {
      repeated_positions += literal_occurrences[i];
      ++repeated_classes;
    }
  }
  for (std::uint32_t i = 0u; i < donor_variables.count; ++i) {
    if (donor_occurrences[i] >= 2u) {
      repeated_positions += donor_occurrences[i];
      ++repeated_classes;
    }
  }
  if (repeated_positions < 2u || repeated_classes < 2u) return false;
  // The pair tables are discovery workspaces, not resident variable identity.
  // Number every resulting class by its first term occurrence so the pattern
  // written to Records is identical to cross_context_pattern_from_program().
  // Without this final canonicalization, a higher-order product whose new
  // literal class appears around inherited variables writes a digest that its
  // own live Record topology cannot reproduce.
  std::uint32_t next_canonical = 0u;
  for (std::uint32_t i = 0u; i < pattern.term_count; ++i) {
    if (variable_kind[i] == 0u) {
      pattern.term_meta[i] = 0u;
      continue;
    }
    const std::uint32_t discovered =
        variable_kind[i] == 1u
            ? variable_index[i]
            : changing_literals.count + variable_index[i];
    std::uint32_t canonical = kInvalid;
    for (std::uint32_t prior = 0u; prior < i; ++prior) {
      if (variable_kind[prior] == 0u) continue;
      const std::uint32_t prior_discovered =
          variable_kind[prior] == 1u
              ? variable_index[prior]
              : changing_literals.count + variable_index[prior];
      if (prior_discovered == discovered) {
        canonical = pattern.term_meta[prior] - 1u;
        break;
      }
    }
    if (canonical == kInvalid) canonical = next_canonical++;
    pattern.term_meta[i] = canonical + 1u;
    std::uint32_t value = 0u, ignored_meta = 0u;
    if (!program_term_at(state, left.lane[1], i, &value, &ignored_meta))
      return false;
    pattern.term_value[i] = value & kRawChannelMask;
  }
  if (next_canonical != pattern.variable_count) return false;
  pattern.digest = cross_context_pattern_digest(pattern);
  *out = pattern;
  return true;
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t cross_context_find_program(
    const ResidentRewriteState* state,
    const CrossContextProgramPattern& wanted) {
  if (state == nullptr) return kInvalid;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& candidate = state->records[i];
    CrossContextProgramPattern existing{};
    if (candidate.matter_q8 != 0u && candidate.lane[0] == kFormProgram &&
        (candidate.lane[7] & kProgramFlagVersionSpace) == 0u &&
        cross_context_pattern_from_program(state, candidate, &existing) &&
        cross_context_pattern_equal(existing, wanted))
      return i;
  }
  return kInvalid;
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t cross_context_program_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr) return kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (program.matter_q8 != 0u && program.lane[0] == kFormProgram &&
        (program.lane[7] & kProgramFlagVersionSpace) == 0u &&
        program.lane[1] == owner)
      return slot;
  }
  return kInvalid;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_witness_form(
    const Record& record) {
  return record.matter_q8 != 0u && record.lane[0] == kFormTransformationWitness;
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t
cross_context_transformation_depth(const ResidentRewriteState* state,
                                   std::uint32_t program_slot);

#if defined(__CUDACC__)
static BCC32_CROSS_CONTEXT_HD __noinline__ bool
cross_context_program_is_derived_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot);
#else
BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_is_derived_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot);
#endif

BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_has_witnesses(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const std::uint32_t owner = state->records[program_slot].lane[1];
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    if (cross_context_witness_form(state->records[i]) &&
        state->records[i].lane[1] == owner)
      return true;
  return false;
}

BCC32_CROSS_CONTEXT_DISPATCH bool cross_context_witness_valid(
    const ResidentRewriteState* state, std::uint32_t witness_slot,
    std::uint32_t product_slot) {
  if (state == nullptr || witness_slot >= live_record_capacity(state) ||
      product_slot >= live_record_capacity(state))
    return false;
  const Record& witness = state->records[witness_slot];
  const Record& product = state->records[product_slot];
  if (!cross_context_witness_form(witness) ||
      product.matter_q8 == 0u || product.lane[0] != kFormProgram ||
      (product.lane[7] & kProgramFlagVersionSpace) != 0u ||
      witness.lane[1] != product.lane[1] || witness.lane[7] == 0u ||
      witness.lane[4] == product.lane[1])
    return false;
  CrossContextProgramPattern product_pattern{}, donor_pattern{};
  const std::uint32_t donor_slot =
      cross_context_program_by_owner(state, witness.lane[4]);
  if (donor_slot == kInvalid ||
      !cross_context_pattern_from_program(state, product, &product_pattern) ||
      !cross_context_pattern_from_program(state, state->records[donor_slot],
                                          &donor_pattern))
    return false;
  if (witness.lane[3] != donor_pattern.digest ||
      witness.lane[5] != product_pattern.digest ||
      witness.lane[2] >= kMaximumCrossContextDepth)
    return false;
  const std::uint32_t donor_depth =
      cross_context_transformation_depth(state, donor_slot);
  if (donor_depth != witness.lane[2])
    return false;
  const bool donor_is_derived =
      cross_context_program_has_witnesses(state, donor_slot);
  return donor_is_derived ? donor_depth != 0u : donor_depth == 0u;
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t
cross_context_transformation_depth(const ResidentRewriteState* state,
                                   std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return kInvalid;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      (program.lane[7] & kProgramFlagVersionSpace) != 0u)
    return kInvalid;
  std::uint32_t depth = kInvalid;
  bool found = false;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& witness = state->records[i];
    if (!cross_context_witness_form(witness) || witness.lane[1] != program.lane[1])
      continue;
    if (witness.lane[2] >= kMaximumCrossContextDepth ||
        witness.lane[7] == 0u)
      return kInvalid;
    if (!found) {
      depth = witness.lane[2];
      found = true;
    } else if (depth != witness.lane[2]) {
      return kInvalid;
    }
  }
  if (!found) return 0u;
  return depth < kMaximumCrossContextDepth ? depth + 1u : kInvalid;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_is_derived(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  return cross_context_program_is_derived_exact(state, program_slot);
}

// Validate one derived node without following derived donors. The complete
// lineage walk below composes this bounded predicate iteratively so CUDA never
// needs a recursive device call graph.
BCC32_CROSS_CONTEXT_DISPATCH bool cross_context_derived_node_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& product = state->records[program_slot];
  if (product.matter_q8 == 0u || product.lane[0] != kFormProgram ||
      (product.lane[7] & kProgramFlagVersionSpace) != 0u)
    return false;
  CrossContextProgramPattern product_pattern{};
  if (!cross_context_pattern_from_program(state, product, &product_pattern))
    return false;
  const std::uint32_t product_depth =
      cross_context_transformation_depth(state, program_slot);
  if (product_depth == kInvalid || product_depth == 0u ||
      product_depth > kMaximumCrossContextDepth)
    return false;
  std::uint32_t contributor_slots[kMaximumCrossContextContributors]{};
  std::uint32_t contributor_count = 0u;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& witness = state->records[i];
    if (!cross_context_witness_form(witness) ||
        witness.lane[1] != product.lane[1])
      continue;
    if (!cross_context_witness_valid(state, i, program_slot) ||
        witness.lane[2] + 1u != product_depth)
      return false;
    const std::uint32_t donor =
        cross_context_program_by_owner(state, witness.lane[4]);
    bool duplicate_owner = false;
    for (std::uint32_t j = 0u; j < contributor_count; ++j)
      duplicate_owner |= state->records[contributor_slots[j]].lane[1] ==
                         state->records[donor].lane[1];
    if (!duplicate_owner) {
      if (contributor_count >= kMaximumCrossContextContributors) return false;
      contributor_slots[contributor_count++] = donor;
    }
  }
  if (contributor_count < 2u || product.lane[3] != contributor_count)
    return false;
  CrossContextProgramPattern first{};
  if (!cross_context_pattern_from_program(
          state, state->records[contributor_slots[0]], &first))
    return false;
  for (std::uint32_t i = 1u; i < contributor_count; ++i) {
    CrossContextProgramPattern candidate{};
    if (!cross_context_derive_pattern(
            state, state->records[contributor_slots[0]],
            state->records[contributor_slots[i]], &candidate) ||
        !cross_context_pattern_equal(candidate, product_pattern))
      return false;
  }
  return true;
}

// A digest identifies no authority. A product remains derived only while its
// complete contributor DAG can reproduce every exact anti-unifier. Declared
// depth must decrease at each edge; repeated DAG nodes are shared, while a
// cycle, malformed depth, or lineage wider than the bounded executor abstains.
BCC32_CROSS_CONTEXT_DISPATCH bool cross_context_program_is_derived_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      !cross_context_program_has_witnesses(state, program_slot))
    return false;

  std::uint32_t queue[kMaximumCrossContextLineage]{};
  std::uint32_t seen_owner[kMaximumCrossContextLineage]{};
  std::uint32_t queue_count = 1u;
  std::uint32_t cursor = 0u;
  std::uint32_t seen_count = 1u;
  queue[0] = program_slot;
  seen_owner[0] = state->records[program_slot].lane[1];

  while (cursor < queue_count) {
    const std::uint32_t current_slot = queue[cursor++];
    if (!cross_context_derived_node_exact(state, current_slot)) return false;
    const Record& current = state->records[current_slot];
    const std::uint32_t current_depth =
        cross_context_transformation_depth(state, current_slot);
    for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
      const Record& witness = state->records[i];
      if (!cross_context_witness_form(witness) ||
          witness.lane[1] != current.lane[1])
        continue;
      const std::uint32_t donor_slot =
          cross_context_program_by_owner(state, witness.lane[4]);
      if (donor_slot == kInvalid) return false;
      const bool donor_is_derived =
          cross_context_program_has_witnesses(state, donor_slot);
      if (!donor_is_derived) continue;
      const std::uint32_t donor_depth =
          cross_context_transformation_depth(state, donor_slot);
      if (current_depth == kInvalid || donor_depth == kInvalid ||
          donor_depth >= current_depth)
        return false;
      const std::uint32_t donor_owner = state->records[donor_slot].lane[1];
      bool seen = false;
      for (std::uint32_t j = 0u; j < seen_count; ++j)
        seen |= seen_owner[j] == donor_owner;
      if (seen) continue;
      if (queue_count >= kMaximumCrossContextLineage ||
          seen_count >= kMaximumCrossContextLineage)
        return false;
      queue[queue_count++] = donor_slot;
      seen_owner[seen_count++] = donor_owner;
    }
  }
  return true;
}

// Once an authoritative product carries a Program as exact construction
// evidence, that donor remains resident and executable but leaves the active
// factoring frontier. Recombining every old donor with every later family
// would manufacture a combinatorial cloud of cross-family anti-unifiers and
// prevent a later coherent family from reaching consensus. The product is the
// reusable structural representative; focal loss of that product immediately
// exposes its grounded donors to ordinary refactoring again.
BCC32_CROSS_CONTEXT_DISPATCH bool cross_context_program_is_subordinated_donor(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const std::uint32_t owner = state->records[program_slot].lane[1];
  for (std::uint32_t witness_slot = 0u; witness_slot < live_record_capacity(state);
       ++witness_slot) {
    const Record& witness = state->records[witness_slot];
    if (!cross_context_witness_form(witness) || witness.lane[4] != owner)
      continue;
    const std::uint32_t product_slot =
        cross_context_program_by_owner(state, witness.lane[1]);
    if (product_slot != kInvalid &&
        (state->records[product_slot].lane[7] & kProgramFlagEnabled) != 0u &&
        cross_context_program_is_derived_exact(state, product_slot))
      return true;
  }
  return false;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_is_eligible_donor(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      (program.lane[7] & kProgramFlagVersionSpace) != 0u ||
      !resident_program_authoritative(state, program_slot))
    return false;
  if ((program.lane[7] & kProgramFlagPureExternalExact) != 0u &&
      (program.lane[3] <= kProgramMatureSupport || program.lane[6] == 0u ||
       program.lane[6] == kInvalid ||
       program.lane[6] == program.lane[1]))
    return false;
  if (program.lane[2] == 0u ||
      program.lane[2] > kMaximumCrossContextTerms ||
      program.lane[4] > kMaximumProgramVariables ||
      cross_context_transformation_depth(state, program_slot) == kInvalid ||
      cross_context_program_is_subordinated_donor(state, program_slot))
    return false;
  bool has_witness = false;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    has_witness |= cross_context_witness_form(state->records[i]) &&
                   state->records[i].lane[1] == program.lane[1];
  return !has_witness || cross_context_program_is_derived(state, program_slot);
}

BCC32_CROSS_CONTEXT_HD inline std::uint32_t cross_context_witness_for_donor(
    const ResidentRewriteState* state, std::uint32_t product_slot,
    std::uint32_t donor_owner, const CrossContextProgramPattern& donor_pattern) {
  if (state == nullptr || product_slot >= live_record_capacity(state)) return kInvalid;
  const std::uint32_t product_owner = state->records[product_slot].lane[1];
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& witness = state->records[i];
    if (!cross_context_witness_form(witness) || witness.lane[1] != product_owner)
      continue;
    const std::uint32_t donor_slot =
        cross_context_program_by_owner(state, witness.lane[4]);
    CrossContextProgramPattern existing{};
    if (donor_slot != kInvalid && cross_context_witness_valid(
                                     state, i, product_slot) &&
        cross_context_pattern_from_program(
                                     state, state->records[donor_slot], &existing) &&
        cross_context_pattern_equal(existing, donor_pattern))
      return i;
  }
  (void)donor_owner;
  return kInvalid;
}

// Transactional installation: every owner, term, witness, and free slot is
// checked before the first resident Record is changed.
BCC32_CROSS_CONTEXT_HD inline bool cross_context_install_product(
    ResidentRewriteState* state, const CrossContextProgramPattern& product,
    const std::uint32_t* contributor_owners, std::uint32_t contributor_count,
    std::uint32_t donor_depth) {
  if (state == nullptr || contributor_owners == nullptr ||
      contributor_count < 2u || product.term_count == 0u ||
      product.term_count > kMaximumCrossContextTerms ||
      donor_depth >= kMaximumCrossContextDepth ||
      contributor_count > kMaximumCrossContextContributors)
    return false;
  const std::uint32_t blocks = (product.term_count + 1u) / 2u;
  std::uint32_t existing = kInvalid;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    if (!cross_context_program_is_derived(state, i)) continue;
    CrossContextProgramPattern candidate{};
    if (cross_context_pattern_from_program(state, state->records[i], &candidate) &&
        cross_context_pattern_equal(candidate, product)) {
      existing = i;
      break;
    }
  }
  std::uint32_t missing = 0u;
  if (existing != kInvalid) {
    // Equal product topology does not collapse construction levels. A base
    // family may later rediscover the same executable pattern as a depth-two
    // product; attaching depth-zero witnesses to that higher-order node would
    // corrupt its uniform lineage depth and let a new episode self-credit it.
    if (cross_context_transformation_depth(state, existing) != donor_depth + 1u)
      return false;
    for (std::uint32_t i = 0u; i < contributor_count; ++i) {
      CrossContextProgramPattern donor_pattern{};
      const std::uint32_t donor =
          cross_context_program_by_owner(state, contributor_owners[i]);
      if (donor == kInvalid || !cross_context_pattern_from_program(
                                  state, state->records[donor], &donor_pattern))
        return false;
      if (cross_context_witness_for_donor(state, existing, contributor_owners[i],
                                          donor_pattern) == kInvalid)
        ++missing;
    }
    if (free_record_count(state) < missing) return false;
  } else {
    if (free_record_count(state) < blocks + 1u + contributor_count) return false;
  }
  const std::uint32_t planned_owner =
      existing == kInvalid
          ? make_record_owner(state, product.digest ^
                                  rewrite_mix(donor_depth, contributor_count,
                                              0x71u))
          : kInvalid;
  if (existing == kInvalid && planned_owner == kInvalid) return false;

  if (existing != kInvalid) {
    Record& program = state->records[existing];
    std::uint32_t added = 0u;
    for (std::uint32_t i = 0u; i < contributor_count; ++i) {
      const std::uint32_t donor =
          cross_context_program_by_owner(state, contributor_owners[i]);
      CrossContextProgramPattern donor_pattern{};
      (void)cross_context_pattern_from_program(state, state->records[donor],
                                               &donor_pattern);
      if (cross_context_witness_for_donor(state, existing, contributor_owners[i],
                                          donor_pattern) != kInvalid)
        continue;
      const std::uint32_t slot = allocate_record(state);
      if (slot == kInvalid) return false;  // preflight makes this unreachable
      Record& witness = state->records[slot];
      witness.lane[0] = kFormTransformationWitness;
      witness.lane[1] = program.lane[1];
      witness.lane[2] = donor_depth;
      witness.lane[3] = donor_pattern.digest;
      witness.lane[4] = contributor_owners[i];
      witness.lane[5] = product.digest;
      witness.lane[6] = 0u;
      witness.lane[7] = 1u;
      ++witness.revision;
      ++added;
    }
    if (added != 0u) {
      program.lane[3] += added;
      ++program.revision;
      ++state->revision;
      ++state->completed_inductions;
      refresh_receipt(state);
    }
    return added != 0u;
  }

  const std::uint32_t header_slot = allocate_record(state);
  const std::uint32_t owner = planned_owner;
  if (header_slot == kInvalid || owner == kInvalid) return false;
  Record& program = state->records[header_slot];
  program.lane[0] = kFormProgram;
  program.lane[1] = owner;
  program.lane[2] = product.term_count;
  program.lane[3] = contributor_count;
  program.lane[4] = product.variable_count;
  program.lane[5] = product.digest;
  program.lane[7] =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly;
  ++program.revision;
  for (std::uint32_t block_index = 0u; block_index < blocks; ++block_index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& term = state->records[slot];
    term.lane[0] = kFormProgramTerm;
    term.lane[1] = owner;
    term.lane[2] = block_index;
    for (std::uint32_t local = 0u; local < 2u; ++local) {
      const std::uint32_t index = block_index * 2u + local;
      if (index >= product.term_count) break;
      term.lane[3u + local * 2u] = product.term_meta[index];
      term.lane[4u + local * 2u] = product.term_value[index];
    }
    ++term.revision;
  }
  for (std::uint32_t i = 0u; i < contributor_count; ++i) {
    const std::uint32_t donor =
        cross_context_program_by_owner(state, contributor_owners[i]);
    CrossContextProgramPattern donor_pattern{};
    (void)cross_context_pattern_from_program(state, state->records[donor],
                                             &donor_pattern);
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& witness = state->records[slot];
    witness.lane[0] = kFormTransformationWitness;
    witness.lane[1] = owner;
    witness.lane[2] = donor_depth;
    witness.lane[3] = donor_pattern.digest;
    witness.lane[4] = contributor_owners[i];
    witness.lane[5] = product.digest;
    witness.lane[6] = 0u;
    witness.lane[7] = 1u;
    ++witness.revision;
  }
  ++state->revision;
  ++state->completed_inductions;
  refresh_receipt(state);
  return true;
}

BCC32_CROSS_CONTEXT_DISPATCH bool
cross_context_factor_mature_program_prevalidated(
    ResidentRewriteState* state, std::uint32_t newcomer_slot,
    bool eligibility_prevalidated,
    const std::uint32_t* eligibility_snapshot = nullptr) {
  if (state == nullptr || newcomer_slot >= live_record_capacity(state))
    return false;
  if (!eligibility_prevalidated &&
      !cross_context_program_is_eligible_donor(state, newcomer_slot))
    return false;
  CrossContextProgramPattern newcomer{};
  if (!cross_context_pattern_from_program(state, state->records[newcomer_slot],
                                          &newcomer))
    return false;
  const std::uint32_t depth =
      cross_context_transformation_depth(state, newcomer_slot);
  if (depth == kInvalid || depth >= kMaximumCrossContextDepth) return false;
  CrossContextProgramPattern consensus{};
  bool have_consensus = false;
  std::uint32_t contributors[kMaximumCrossContextContributors]{};
  std::uint32_t contributor_count = 0u;
  const std::uint32_t minimum_contributors = depth == 0u ? 3u : 2u;
  // Both branches now range over the full live population: the snapshot
  // branch relies on the caller (cross_context_factor_all_mature_programs)
  // having sized its bitmask to live_record_capacity(state), and the
  // fallback branch calls cross_context_program_is_eligible_donor, which
  // already bounds-checks against live_record_capacity(state) itself.
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const bool donor_eligible =
        eligibility_snapshot != nullptr
            ? (eligibility_snapshot[i / 32u] & (1u << (i % 32u))) != 0u
            : cross_context_program_is_eligible_donor(state, i);
    if (i == newcomer_slot || !donor_eligible ||
        cross_context_transformation_depth(state, i) != depth)
      continue;
    CrossContextProgramPattern donor{};
    if (!cross_context_pattern_from_program(state, state->records[i], &donor))
      continue;
    CrossContextProgramPattern candidate{};
    if (!cross_context_derive_pattern(state, state->records[newcomer_slot],
                                      state->records[i], &candidate))
      continue;
    if (!have_consensus) {
      consensus = candidate;
      have_consensus = true;
    } else if (!cross_context_pattern_equal(consensus, candidate)) {
      // Same-depth frontier donors disagree. Do not choose by slot, owner,
      // digest, or support accident; wait for further grounded structure.
      return false;
    }
    bool duplicate = false;
    for (std::uint32_t prior = 0u; prior < contributor_count; ++prior) {
      const std::uint32_t prior_slot =
          cross_context_program_by_owner(state, contributors[prior]);
      CrossContextProgramPattern prior_pattern{};
      if (prior_slot != kInvalid && cross_context_pattern_from_program(
                                      state, state->records[prior_slot],
                                      &prior_pattern) &&
          cross_context_pattern_equal(donor, prior_pattern)) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    if (contributor_count >= kMaximumCrossContextContributors) return false;
    contributors[contributor_count++] = state->records[i].lane[1];
  }
  if (!have_consensus || contributor_count == 0u) return false;
  // The newcomer is itself a contributor; identical structures are still one
  // witness, which keeps support structural rather than owner-count based.
  bool newcomer_duplicate = false;
  for (std::uint32_t i = 0u; i < contributor_count; ++i) {
    CrossContextProgramPattern prior{};
    const std::uint32_t prior_slot =
        cross_context_program_by_owner(state, contributors[i]);
    if (prior_slot != kInvalid && cross_context_pattern_from_program(
                                    state, state->records[prior_slot], &prior) &&
        cross_context_pattern_equal(prior, newcomer)) {
      newcomer_duplicate = true;
      break;
    }
  }
  if (!newcomer_duplicate) {
    if (contributor_count >= kMaximumCrossContextContributors) return false;
    for (std::uint32_t i = contributor_count; i > 0u; --i)
      contributors[i] = contributors[i - 1u];
    contributors[0] = state->records[newcomer_slot].lane[1];
    ++contributor_count;
  }
  // A base-level family must be repeated across three independently mature
  // Programs.  Otherwise an accidental pair spanning two unrelated families
  // can become a depth-one donor before either real family has completed.
  // Higher-order products are different: two already grounded products are
  // sufficient because each carries its own complete witness lineage.
  if (contributor_count < minimum_contributors) return false;
  return cross_context_install_product(state, consensus, contributors,
                                       contributor_count, depth);
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_factor_mature_program(
    ResidentRewriteState* state, std::uint32_t newcomer_slot) {
  return cross_context_factor_mature_program_prevalidated(
      state, newcomer_slot, false);
}

 BCC32_CROSS_CONTEXT_DISPATCH bool
cross_context_factor_all_mature_programs(ResidentRewriteState* state) {
  if (state == nullptr) return false;
  // Most End boundaries have no possible pair.  Keep that hot production path
  // out of the recursive authority/anti-unification machinery entirely.
  std::uint32_t potential_donors = 0u;
  for (std::uint32_t i = 0u; i < live_record_capacity(state) && potential_donors < 2u; ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 != 0u && record.lane[0] == kFormProgram &&
        (record.lane[7] & kProgramFlagVersionSpace) == 0u &&
        record.lane[3] >= kProgramMatureSupport)
      ++potential_donors;
  }
  if (potential_donors < 2u) return false;
  const std::uint32_t capacity = live_record_capacity(state);
  const std::uint32_t words = cross_context_eligibility_words(capacity);
  // Heap-sized to the live population, not a kRecordCapacity-sized stack
  // array (0X1-222): population grows past page 0 via grow_resident_pages,
  // and this bitmask must keep covering every page or donors beyond page 0
  // silently stop being considered. Same host/device dual-context malloc
  // idiom as grow_resident_pages and resolve_relation_step -- this function
  // has one call site inside a single continuing resident thread (see
  // bcc32_resident_mixed_provenance_evidence.cuh), not a per-Record
  // parallel kernel, so one malloc/free pair per call is safe.
  auto* eligible =
      static_cast<std::uint32_t*>(malloc(sizeof(std::uint32_t) * words));
  if (eligible == nullptr) return false;
  bool changed = false;
  // A round is an immutable authority view. Products installed while this
  // view is consumed cannot become their own donors or affect another donor's
  // eligibility until the next bounded physical-END round.
  for (std::uint32_t round = 0u; round < kMaximumCrossContextDepth; ++round) {
    for (std::uint32_t w = 0u; w < words; ++w) eligible[w] = 0u;
    for (std::uint32_t i = 0u; i < capacity; ++i)
      if (cross_context_program_is_eligible_donor(state, i))
        eligible[i / 32u] |= 1u << (i % 32u);

    bool round_changed = false;
    for (std::uint32_t i = 0u; i < capacity; ++i)
      if ((eligible[i / 32u] & (1u << (i % 32u))) != 0u)
        round_changed |=
            cross_context_factor_mature_program_prevalidated(state, i, true,
                                                              eligible);
    changed |= round_changed;
    if (!round_changed) break;
  }
  free(eligible);
  return changed;
}

// Compatibility spelling retained for coordinator wiring; the implementation
// is the generic recursive helper above, not a depth-one marker path.
BCC32_CROSS_CONTEXT_HD inline bool try_induce_cross_context_program(
    ResidentRewriteState* state, std::uint32_t new_base_program_slot) {
  return cross_context_factor_mature_program(state, new_base_program_slot);
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_is_marked(
    const ResidentRewriteState* state, const Record& program) {
  if (state == nullptr) return false;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    if (&state->records[i] == &program)
      return cross_context_program_is_derived(state, i);
  return false;
}

BCC32_CROSS_CONTEXT_HD inline bool cross_context_program_is_mature(
    const ResidentRewriteState* state, std::uint32_t slot) {
  return resident_program_authoritative(state, slot) &&
         cross_context_program_is_derived(state, slot);
}

}  // namespace substrate::bcc32::causal_rewrite::cross_context

namespace substrate::bcc32::causal_rewrite {

// One authority predicate serves every executor and resident observer.  Base
// Programs are grounded by independent raw support.  A resident-evidence-only
// Program additionally remains authoritative only while its exact contributor
// witnesses and live donor structures reproduce the complete anti-unifier.
BCC32_CROSS_CONTEXT_DISPATCH bool resident_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth) {
  if (recursion_depth > kMaxCausalGermlineRecursionDepth) return false;
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
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
      (program.lane[7] & kProgramFlagRevisionTransferProduct) != 0u)
    return resident_revision_transfer_product_authoritative(
        state, program_slot, recursion_depth);
  if (program.lane[0] != kFormProgram)
    return false;
  // Ordinary Programs mature from repeated external support. A derived
  // Program instead carries exact live contributor lineage, so two mature
  // higher-order donors can ground it even though the base support threshold
  // remains three. Do not collapse those distinct authority currencies into
  // lane[3]'s numeric threshold.
  if ((program.lane[7] & kProgramFlagResidentEvidenceOnly) != 0u)
    return cross_context::cross_context_program_is_derived_exact(
        state, program_slot);
  return program.lane[3] >= kProgramMatureSupport &&
         !resident_program_shadowed_by_revision(state, program_slot,
                                                 recursion_depth);
}

}  // namespace substrate::bcc32::causal_rewrite

#undef BCC32_CROSS_CONTEXT_HD
#undef BCC32_CROSS_CONTEXT_DISPATCH
