// Included by bcc32_resident_cross_contact_context.cuh inside
// substrate::bcc32::causal_rewrite::cross_contact, after the RWR11 prefix
// recognizer. This file adds no state owner or Record form. It only changes
// how the existing Carry trajectory phenotype survives and rejoins later raw
// contact.

inline constexpr std::uint32_t kMaximumJoinProbeRecords =
 1u + (causal_rewrite::kMaximumTrajectoryEvents + 1u) / 2u;
// Dormant discourse is zero-authority context and may consume otherwise free
// Records, but it may not starve the ordinary current-contact/join machinery.
// This is a physical Record reserve, not a memory-count ceiling.
inline constexpr std::uint32_t kDormantOperationalReserve =
 kMaximumJoinProbeRecords +
 (causal_rewrite::kMaximumTrajectoryEvents + 1u) / 2u + 3u;

// Local zero-authority Carry representation bit; no Program or evidence path
// accepts it as raw teaching matter.
inline constexpr std::uint32_t kDormantCompactCarry = 1u << 5u;
inline constexpr std::uint32_t kCompactLengthsPerWord = 4u;
inline constexpr std::uint32_t kCompactLengthBits = 7u;
inline constexpr std::uint32_t kCompactLengthMask =
 (1u << kCompactLengthBits) - 1u;
inline constexpr std::uint32_t kMaximumCompactLengthWords =
 (causal_rewrite::kMaximumProgramVariables + kCompactLengthsPerWord - 1u) /
 kCompactLengthsPerWord;
inline constexpr std::uint32_t kCompactMetadataWords = 4u;
inline constexpr std::uint32_t kDormantCompactFormatMagic = 0x43445031u;
// This bounded fact is covered by the compact-history digests. It is not a
// retained provenance Record and can become ordinary provenance again only
// when a fresh wholly-external contact joins the carry.
inline constexpr std::uint32_t kDormantCompactExternalProvenance = 1u << 31u;
inline constexpr std::uint32_t kDormantCompactMetadataFlagMask = 3u << 30u;
inline constexpr std::uint32_t kDormantCompactPayloadWordsMask =
    ~kDormantCompactMetadataFlagMask;
inline constexpr std::uint32_t kMaximumCompactPayloadWords =
 causal_rewrite::kMaximumTrajectoryEvents;
inline constexpr std::uint32_t kMaximumCompactProgramTerms =
 causal_rewrite::kMaximumSpanProgramTerms;
static_assert(causal_rewrite::kMaximumVariableSpanEvents <=
              kCompactLengthMask);

struct CompactDormantPlan {
 std::uint32_t term_count = 0u;
 std::uint32_t variable_count = 0u;
 std::uint32_t payload_words = 0u;
 std::uint32_t identity = 0u;
 std::uint32_t length_words = 0u;
 std::uint32_t binding_offset = 0u;
 std::uint32_t format_magic = 0u;
 std::uint32_t kind_words = 0u;
 std::uint32_t values_offset = 0u;
 bool external_provenance = false;
 std::uint32_t payload_slot[(kMaximumCompactPayloadWords + 1u) / 2u]{};
 std::uint32_t length[causal_rewrite::kMaximumProgramVariables]{};
 std::uint32_t binding_base[causal_rewrite::kMaximumProgramVariables]{};
};

BCC32_CROSS_CONTACT_HD inline bool is_dormant_history(
 const Record& trajectory) {
 return trajectory.matter_q8 != 0u &&
 trajectory.lane[0] == causal_rewrite::kFormTrajectory &&
 trajectory.lane[3] != 0u && is_carried_history(trajectory);
}

BCC32_CROSS_CONTACT_HD inline bool is_compact_dormant_history(
 const Record& trajectory) {
 return is_dormant_history(trajectory) &&
        trajectory.lane[7] ==
            (kTrajectoryHasCarry | kDormantCompactCarry);
}

BCC32_CROSS_CONTACT_HD inline bool trajectory_is_completely_external(
 const ResidentRewriteState* state, const Record& trajectory);
BCC32_CROSS_CONTACT_HD inline bool trajectory_has_provenance(
 const ResidentRewriteState* state, const Record& trajectory);
BCC32_CROSS_CONTACT_HD inline bool trajectory_has_complete_provenance(
 const ResidentRewriteState* state, const Record& trajectory);

BCC32_CROSS_CONTACT_HD inline bool unique_owned_block_slot(
 const ResidentRewriteState* state, std::uint32_t form,
 std::uint32_t owner, std::uint32_t ordinal,
 std::uint32_t* unique_slot) {
 if (state == nullptr || unique_slot == nullptr) return false;
 *unique_slot = causal_rewrite::kInvalid;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  const Record& record = state->records[slot];
  if (record.matter_q8 == 0u || record.lane[0] != form ||
      record.lane[1] != owner || record.lane[2] != ordinal)
   continue;
  if (*unique_slot != causal_rewrite::kInvalid) return false;
  *unique_slot = slot;
 }
 return *unique_slot != causal_rewrite::kInvalid;
}

BCC32_CROSS_CONTACT_HD inline bool owned_block_topology_exact(
 const ResidentRewriteState* state, std::uint32_t form,
 std::uint32_t owner, std::uint32_t expected_blocks) {
 if (state == nullptr || expected_blocks == 0u ||
     expected_blocks > causal_rewrite::live_record_capacity(state))
  return false;
 std::uint32_t owned = 0u;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  const Record& record = state->records[slot];
  if (record.matter_q8 == 0u || record.lane[0] != form ||
      record.lane[1] != owner)
   continue;
  if (record.lane[2] >= expected_blocks) return false;
  ++owned;
 }
 if (owned != expected_blocks) return false;
 for (std::uint32_t ordinal = 0u; ordinal < expected_blocks; ++ordinal) {
  std::uint32_t ignored = causal_rewrite::kInvalid;
  if (!unique_owned_block_slot(state, form, owner, ordinal, &ignored))
   return false;
 }
 return true;
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t compact_span_digest(
 const ResidentRewriteState* state, std::uint32_t slot) {
 if (state == nullptr) return 0u;
 if (slot >= causal_rewrite::live_record_capacity(state) ||
     !causal_rewrite::raw_span_program_preflight(state, slot))
  return 0u;
 return causal_rewrite::raw_span_program_identity(state, slot);
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t compact_history_digest(
 std::uint32_t program_digest, std::uint32_t packed_lengths,
 std::uint32_t extent, const std::uint32_t* payload,
 std::uint32_t payload_words) {
 if (program_digest == 0u || payload == nullptr || payload_words == 0u)
  return 0u;
 std::uint32_t digest = causal_rewrite::rewrite_mix(
     program_digest, packed_lengths, extent);
 for (std::uint32_t index = 0u; index < payload_words; ++index)
  digest = causal_rewrite::rewrite_mix(digest, payload[index], index);
 return digest;
}

BCC32_CROSS_CONTACT_HD inline bool compact_payload_word_at(
 const ResidentRewriteState* state, const Record& trajectory,
 std::uint32_t index, std::uint32_t* word) {
 if (state == nullptr || word == nullptr ||
     !is_compact_dormant_history(trajectory))
  return false;
 std::uint32_t block = causal_rewrite::kInvalid;
 if (!unique_owned_block_slot(
         state, causal_rewrite::kFormTrajectoryTerm,
         trajectory.lane[1], index / 2u, &block))
  return false;
 *word = state->records[block].lane[4u + (index % 2u)];
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool compact_plan_payload_word_at(
 const ResidentRewriteState* state, const CompactDormantPlan& plan,
 std::uint32_t index, std::uint32_t* word) {
 if (state == nullptr || word == nullptr || index >= plan.payload_words)
  return false;
 const std::uint32_t slot = plan.payload_slot[index / 2u];
 if (slot >= causal_rewrite::live_record_capacity(state)) return false;
 *word = state->records[slot].lane[4u + index % 2u];
 return true;
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t compact_plan_history_digest(
 const ResidentRewriteState* state, const CompactDormantPlan& plan,
 std::uint32_t identity, std::uint32_t packed_lengths,
 std::uint32_t extent) {
 std::uint32_t digest = causal_rewrite::rewrite_mix(
     identity, packed_lengths, extent);
 for (std::uint32_t index = 0u; index < plan.payload_words; ++index) {
  std::uint32_t word = 0u;
  if (!compact_plan_payload_word_at(state, plan, index, &word)) return 0u;
  digest = causal_rewrite::rewrite_mix(digest, word, index);
 }
 return digest;
}

BCC32_CROSS_CONTACT_HD inline bool compact_plan_term_at(
 const ResidentRewriteState* state, const CompactDormantPlan& plan,
 std::uint32_t term, std::uint32_t* kind, std::uint32_t* value,
 std::uint32_t* channel) {
 if (state == nullptr || kind == nullptr || value == nullptr ||
     channel == nullptr || term >= plan.term_count)
  return false;
 std::uint32_t kinds = 0u;
 std::uint32_t encoded = 0u;
 if (!compact_plan_payload_word_at(
         state, plan, kCompactMetadataWords + term / 32u, &kinds) ||
     !compact_plan_payload_word_at(
         state, plan, plan.values_offset + term, &encoded))
  return false;
 *kind = (kinds & (1u << (term % 32u))) != 0u
             ? causal_rewrite::kSpanTermVariable
             : causal_rewrite::kSpanTermLiteral;
 if (*kind == causal_rewrite::kSpanTermLiteral) {
  *value = encoded;
  *channel = encoded & causal_rewrite::kRawChannelMask;
 } else {
  *value = encoded & 0xffu;
  *channel = encoded & causal_rewrite::kRawChannelMask;
 }
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool build_compact_dormant_plan(
 const ResidentRewriteState* state, const Record& trajectory,
 CompactDormantPlan* plan) {
 if (state == nullptr || plan == nullptr ||
     !is_compact_dormant_history(trajectory))
  return false;
 *plan = CompactDormantPlan{};
 std::uint32_t shape = 0u;
 std::uint32_t encoded_payload_words = 0u;
 if (!compact_payload_word_at(state, trajectory, 0u, &plan->format_magic) ||
     !compact_payload_word_at(state, trajectory, 1u, &plan->identity) ||
     !compact_payload_word_at(state, trajectory, 2u, &shape) ||
     !compact_payload_word_at(state, trajectory, 3u,
                              &encoded_payload_words))
  return false;
 plan->payload_words =
     encoded_payload_words & kDormantCompactPayloadWordsMask;
 const std::uint32_t metadata_flags =
     encoded_payload_words & kDormantCompactMetadataFlagMask;
 plan->term_count = shape >> 16u;
 plan->variable_count = shape & 0xffffu;
 if (plan->format_magic != kDormantCompactFormatMagic ||
     plan->term_count == 0u ||
     plan->term_count > kMaximumCompactProgramTerms ||
     plan->variable_count == 0u ||
     plan->variable_count > causal_rewrite::kMaximumProgramVariables ||
     plan->payload_words > kMaximumCompactPayloadWords ||
     (metadata_flags != 0u &&
      metadata_flags != kDormantCompactExternalProvenance))
  return false;
 plan->external_provenance =
     (encoded_payload_words & kDormantCompactExternalProvenance) != 0u;
 plan->length_words =
     (plan->variable_count + kCompactLengthsPerWord - 1u) /
     kCompactLengthsPerWord;
 plan->kind_words = (plan->term_count + 31u) / 32u;
 plan->values_offset = kCompactMetadataWords + plan->kind_words;
 const std::uint32_t lengths_offset = plan->values_offset + plan->term_count;
 plan->binding_offset = lengths_offset + plan->length_words;
 if (trajectory.lane[5] != plan->length_words ||
     plan->binding_offset > plan->payload_words)
  return false;

 const std::uint32_t blocks = (plan->payload_words + 1u) / 2u;
 if (!owned_block_topology_exact(
         state, causal_rewrite::kFormTrajectoryTerm,
         trajectory.lane[1], blocks))
  return false;
 for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
  if (!unique_owned_block_slot(
          state, causal_rewrite::kFormTrajectoryTerm,
          trajectory.lane[1], ordinal, &plan->payload_slot[ordinal]))
   return false;
  const Record& term = state->records[plan->payload_slot[ordinal]];
  if (term.lane[3] != 0u || term.lane[6] != 0u || term.lane[7] != 0u ||
      term.reserved[0] != 0u || term.reserved[1] != 0u)
   return false;
 }

 std::uint32_t binding_words = 0u;
 for (std::uint32_t variable = 0u; variable < plan->variable_count;
      ++variable) {
  std::uint32_t packed = 0u;
  if (!compact_plan_payload_word_at(
          state, *plan,
          lengths_offset + variable / kCompactLengthsPerWord, &packed))
   return false;
  plan->length[variable] =
      (packed >> ((variable % kCompactLengthsPerWord) * kCompactLengthBits)) &
      kCompactLengthMask;
  if (plan->length[variable] == 0u ||
      plan->length[variable] >
          causal_rewrite::kMaximumVariableSpanEvents)
   return false;
  plan->binding_base[variable] = plan->binding_offset + binding_words;
  binding_words += plan->length[variable];
  if (binding_words > causal_rewrite::kMaximumTrajectoryEvents)
   return false;
 }
 if (plan->binding_offset + binding_words != plan->payload_words)
  return false;
 std::uint32_t expanded = 0u;
 for (std::uint32_t term = 0u; term < plan->term_count; ++term) {
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t ignored = 0u;
 if (!compact_plan_term_at(
          state, *plan, term, &kind, &value, &ignored))
   return false;
  if (kind == causal_rewrite::kSpanTermVariable &&
      value >= plan->variable_count)
   return false;
  if (expanded < trajectory.lane[2])
   expanded += kind == causal_rewrite::kSpanTermLiteral
                   ? 1u
                   : plan->length[value];
 }
 // A Carry is deliberately an incomplete matched prefix. The retained extent
 // can end before later stored terms; it must only be covered exactly by the
 // canonical expansion, never require the whole Program to have completed.
 if (expanded < trajectory.lane[2]) return false;
 const std::uint32_t final_used =
     ((plan->variable_count - 1u) % kCompactLengthsPerWord + 1u) *
     kCompactLengthBits;
 std::uint32_t final_lengths = 0u;
 if (!compact_plan_payload_word_at(
         state, *plan, lengths_offset + plan->length_words - 1u,
         &final_lengths) ||
     (final_used < 32u && (final_lengths >> final_used) != 0u))
  return false;
 if ((plan->payload_words & 1u) != 0u) {
  std::uint32_t final_slot = causal_rewrite::kInvalid;
  if (!unique_owned_block_slot(
          state, causal_rewrite::kFormTrajectoryTerm,
          trajectory.lane[1], blocks - 1u, &final_slot) ||
      state->records[final_slot].lane[5] != 0u)
   return false;
 }
 const std::uint32_t primary = compact_plan_history_digest(
     state, *plan, plan->identity, plan->length_words, trajectory.lane[2]);
 const std::uint32_t secondary = compact_plan_history_digest(
     state, *plan,
     plan->identity ^ 0x9e3779b9u, plan->length_words,
     trajectory.lane[2] ^ 0x85ebca6bu);
 const std::uint32_t tertiary = compact_plan_history_digest(
     state, *plan,
     plan->identity ^ 0xc2b2ae35u,
     plan->length_words ^ 0x27d4eb2du, trajectory.lane[2]);
 return primary != 0u && primary == trajectory.lane[6] &&
        secondary == trajectory.reserved[0] &&
        tertiary == trajectory.reserved[1];
}

BCC32_CROSS_CONTACT_HD inline bool compact_word_from_plan(
 const ResidentRewriteState* state, const Record& trajectory,
 const CompactDormantPlan& plan, std::uint32_t index,
 std::uint32_t* word) {
 if (state == nullptr || word == nullptr || index >= trajectory.lane[2])
  return false;
 std::uint32_t logical = 0u;
 for (std::uint32_t term = 0u; term < plan.term_count; ++term) {
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t channel = 0u;
  if (!compact_plan_term_at(
          state, plan, term, &kind, &value, &channel))
   return false;
  if (kind == causal_rewrite::kSpanTermLiteral) {
   if (logical == index) {
    *word = value;
    return true;
   }
   ++logical;
   continue;
  }
  const std::uint32_t variable = value;
  const std::uint32_t length = plan.length[variable];
  if (index >= logical && index < logical + length)
   return compact_plan_payload_word_at(
       state, plan, plan.binding_base[variable] + index - logical, word);
  logical += length;
 }
 return false;
}

BCC32_CROSS_CONTACT_OUTLINE bool compact_program_and_lengths(
 const ResidentRewriteState* state, const Record& trajectory,
 std::uint32_t* program_slot,
 std::uint32_t lengths[causal_rewrite::kMaximumProgramVariables]) {
 if (state == nullptr || program_slot == nullptr ||
     !is_compact_dormant_history(trajectory))
  return false;
 CompactDormantPlan plan{};
 if (!build_compact_dormant_plan(state, trajectory, &plan)) return false;
 *program_slot = causal_rewrite::kInvalid;
 std::uint32_t matches = 0u;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  const Record& program = state->records[slot];
  if (program.matter_q8 == 0u ||
      program.lane[0] != causal_rewrite::kFormSpanProgram ||
      !causal_rewrite::resident_program_authoritative(state, slot) ||
      !causal_rewrite::raw_span_program_preflight(state, slot) ||
      program.lane[2] != plan.term_count ||
      program.lane[4] != plan.variable_count ||
      program.lane[5] != plan.identity)
   continue;
  bool exact = true;
  for (std::uint32_t term = 0u; term < plan.term_count; ++term) {
   std::uint32_t stored_kind = 0u;
   std::uint32_t stored_value = 0u;
   std::uint32_t stored_channel = 0u;
   std::uint32_t kind = 0u;
   std::uint32_t value = 0u;
   std::uint32_t channel = 0u;
   std::uint32_t reserved = 0u;
   if (!compact_plan_term_at(
           state, plan, term, &stored_kind, &stored_value,
           &stored_channel) ||
       !causal_rewrite::span_program_term_at(
           state, program.lane[1], term, &kind, &value, &channel,
           &reserved) ||
       reserved != 0u || kind != stored_kind || value != stored_value ||
       channel != stored_channel) {
    exact = false;
    break;
   }
  }
  if (!exact) continue;
  *program_slot = slot;
  ++matches;
 }
 if (matches != 1u) {
  *program_slot = causal_rewrite::kInvalid;
  return false;
 }
 for (std::uint32_t variable = 0u; variable < plan.variable_count;
      ++variable)
  lengths[variable] = plan.length[variable];
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool compact_dormant_word_at(
 const ResidentRewriteState* state, const Record& trajectory,
 std::uint32_t index, std::uint32_t* word) {
 if (state == nullptr || word == nullptr || index >= trajectory.lane[2])
  return false;
 if (!is_compact_dormant_history(trajectory))
  return causal_rewrite::trajectory_word_at(
      state, trajectory.lane[1], index, word);
 CompactDormantPlan plan{};
 return build_compact_dormant_plan(state, trajectory, &plan) &&
        compact_word_from_plan(state, trajectory, plan, index, word);
}

// Compaction carries bounded matching/binding workspaces. Keep them out of the
// persistent graph root's frame; this is only a CUDA placement boundary over
// the same ResidentRewriteState transaction.
BCC32_CROSS_CONTACT_OUTLINE bool compact_dormant_history(
 ResidentRewriteState* state, std::uint32_t slot) {
 if (state == nullptr || slot >= causal_rewrite::live_record_capacity(state) ||
     !is_dormant_history(state->records[slot]) ||
     is_compact_dormant_history(state->records[slot]))
  return false;
 Record& trajectory = state->records[slot];
 const std::uint32_t owner = trajectory.lane[1];
 std::uint32_t dormant_headers = 0u;
 for (std::uint32_t candidate = 0u;
      candidate < causal_rewrite::live_record_capacity(state); ++candidate)
  if (is_dormant_history(state->records[candidate]) &&
      state->records[candidate].lane[1] == owner)
   ++dormant_headers;
 if (dormant_headers != 1u) return false;
 const std::uint32_t raw_blocks = (trajectory.lane[2] + 1u) / 2u;
 const bool complete_external_provenance =
     trajectory_is_completely_external(state, trajectory);
 const bool has_provenance = trajectory_has_provenance(state, trajectory);
 // Compact representation may retain a verified mixed/generated trajectory as
 // zero-authority context, but only a completely external source earns the
 // compact external fact. Partial or malformed provenance never compacts.
 if (has_provenance &&
     !trajectory_has_complete_provenance(state, trajectory))
  return false;

 SpanPrefixWitness witness{};
 if (!find_unique_bound_span_prefix_impl(
         state, slot, &witness, true, true) ||
     witness.program_slot >= causal_rewrite::live_record_capacity(state))
  return false;
 const Record& program = state->records[witness.program_slot];
 if (!causal_rewrite::resident_program_authoritative(
         state, witness.program_slot) ||
     program.lane[4] == 0u ||
     program.lane[4] > causal_rewrite::kMaximumProgramVariables)
  return false;
 const std::uint32_t program_digest =
     compact_span_digest(state, witness.program_slot);
 if (program_digest == 0u) return false;

 std::uint32_t bindings[causal_rewrite::kMaximumProgramVariables]
                       [causal_rewrite::kMaximumVariableSpanEvents]{};
 std::uint32_t lengths[causal_rewrite::kMaximumProgramVariables]{};
 std::uint32_t next_term = causal_rewrite::kInvalid;
 std::uint32_t next_offset = 0u;
 bool next_unbound = false;
 bool complete = false;
 bool ambiguous = false;
 const std::uint32_t match = causal_rewrite::span_match_prefix_at(
     state, program, trajectory, 0u, &next_term, &next_offset,
     &next_unbound, &complete, &ambiguous, bindings, lengths);
 if (match != causal_rewrite::kSpanMatchPrefix ||
     next_unbound || ambiguous || complete)
  return false;

 std::uint32_t binding_words = 0u;
 for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable) {
  if (lengths[variable] == 0u ||
      lengths[variable] > causal_rewrite::kMaximumVariableSpanEvents)
   return false;
  binding_words += lengths[variable];
 }
 const std::uint32_t length_words =
     (program.lane[4] + kCompactLengthsPerWord - 1u) /
     kCompactLengthsPerWord;
 const std::uint32_t kind_words = (program.lane[2] + 31u) / 32u;
 const std::uint32_t compact_words =
     kCompactMetadataWords + kind_words + program.lane[2] +
     length_words + binding_words;
 const std::uint32_t compact_blocks = (compact_words + 1u) / 2u;
 if (binding_words > causal_rewrite::kMaximumTrajectoryEvents ||
     compact_words > kMaximumCompactPayloadWords ||
     compact_blocks >= raw_blocks ||
     !owned_block_topology_exact(
         state, causal_rewrite::kFormTrajectoryTerm, owner, raw_blocks) ||
     (complete_external_provenance &&
      !owned_block_topology_exact(
          state, causal_rewrite::kFormTrajectoryProvenance,
          owner, raw_blocks)))
  return false;

 std::uint32_t payload[kMaximumCompactPayloadWords]{};
 payload[0] = kDormantCompactFormatMagic;
 payload[1] = program.lane[5];
 payload[2] = (program.lane[2] << 16u) | program.lane[4];
 payload[3] = compact_words |
              (complete_external_provenance
                   ? kDormantCompactExternalProvenance
                   : 0u);
 const std::uint32_t values_offset = kCompactMetadataWords + kind_words;
 for (std::uint32_t term = 0u; term < program.lane[2]; ++term) {
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  std::uint32_t channel = 0u;
  std::uint32_t reserved = 0u;
  if (!causal_rewrite::span_program_term_at(
          state, program.lane[1], term, &kind, &value, &channel,
          &reserved) ||
      reserved != 0u)
   return false;
  if (kind == causal_rewrite::kSpanTermVariable) {
   payload[kCompactMetadataWords + term / 32u] |=
       1u << (term % 32u);
   payload[values_offset + term] = value | channel;
  } else {
   payload[values_offset + term] = value;
  }
 }
 const std::uint32_t lengths_offset = values_offset + program.lane[2];
 for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable)
  payload[lengths_offset + variable / kCompactLengthsPerWord] |=
      lengths[variable] <<
      ((variable % kCompactLengthsPerWord) * kCompactLengthBits);
 std::uint32_t payload_index = lengths_offset + length_words;
 for (std::uint32_t variable = 0u; variable < program.lane[4]; ++variable)
  for (std::uint32_t offset = 0u; offset < lengths[variable]; ++offset)
   payload[payload_index++] = bindings[variable][offset];
 const std::uint32_t history_digest = compact_history_digest(
     program_digest, length_words, trajectory.lane[2],
     payload, compact_words);
 if (history_digest == 0u) return false;

 std::uint32_t term_slots[causal_rewrite::kMaximumTrajectoryEvents]{};
 for (std::uint32_t ordinal = 0u; ordinal < raw_blocks; ++ordinal) {
  if (!unique_owned_block_slot(
          state, causal_rewrite::kFormTrajectoryTerm,
          owner, ordinal, &term_slots[ordinal]))
   return false;
 }
 if (state->revision == ~0u || trajectory.revision == ~0u)
  return false;
 for (std::uint32_t ordinal = 0u; ordinal < compact_blocks; ++ordinal)
  if (state->records[term_slots[ordinal]].revision == ~0u)
   return false;

 // All source topology and compact payload bytes are validated above. From
 // here the transaction cannot fail: provenance is retired and only the
 // preflighted term slots are rewritten or reclaimed.
 causal_rewrite::clear_owned_records(
     state, causal_rewrite::kFormTrajectoryProvenance, owner);
 for (std::uint32_t ordinal = 0u; ordinal < raw_blocks; ++ordinal) {
  Record& term = state->records[term_slots[ordinal]];
  if (ordinal >= compact_blocks) {
   causal_rewrite::clear_record(&term);
   continue;
  }
  term.lane[2] = ordinal;
  term.lane[3] = 0u;
  const std::uint32_t first = ordinal * 2u;
  term.lane[4] = first < compact_words ? payload[first] : 0u;
  term.lane[5] = first + 1u < compact_words ? payload[first + 1u] : 0u;
  term.lane[6] = 0u;
  term.lane[7] = 0u;
  term.reserved[0] = 0u;
  term.reserved[1] = 0u;
  ++term.revision;
 }
 trajectory.lane[5] = length_words;
 trajectory.lane[6] = history_digest;
 trajectory.lane[7] = kTrajectoryHasCarry | kDormantCompactCarry;
 trajectory.reserved[0] = compact_history_digest(
     program_digest ^ 0x9e3779b9u, length_words,
     trajectory.lane[2] ^ 0x85ebca6bu, payload, compact_words);
 trajectory.reserved[1] = compact_history_digest(
     program_digest ^ 0xc2b2ae35u, length_words ^ 0x27d4eb2du,
     trajectory.lane[2], payload, compact_words);
 ++trajectory.revision;
 ++state->revision;
 return true;
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t dormant_carry_count(
 const ResidentRewriteState* state) {
 if (state == nullptr) return 0u;
 std::uint32_t count = 0u;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot)
 count += is_dormant_history(state->records[slot]);
 return count;
}

BCC32_CROSS_CONTACT_HD inline bool dormant_carry_identity_valid(
 ResidentRewriteState* state, std::uint32_t candidate_slot) {
 if (state == nullptr || candidate_slot >= causal_rewrite::live_record_capacity(state) ||
 !is_dormant_history(state->records[candidate_slot]))
 return false;
 const Record& trajectory = state->records[candidate_slot];
 const std::uint32_t owner = trajectory.lane[1];
 if (owner == 0u || owner == causal_rewrite::kInvalid) return false;
 std::uint32_t owner_headers = 0u;
 for (std::uint32_t slot = 0u;
 slot < causal_rewrite::live_record_capacity(state); ++slot) {
 const Record& record = state->records[slot];
 if (record.matter_q8 != 0u &&
     record.lane[0] == causal_rewrite::kFormTrajectory &&
     record.lane[1] == owner)
 ++owner_headers;
 }
 if (owner_headers != 1u || trajectory.lane[2] == 0u ||
     trajectory.lane[2] > causal_rewrite::kMaximumTrajectoryEvents)
  return false;
 if (is_compact_dormant_history(trajectory)) {
  CompactDormantPlan plan{};
  if (!build_compact_dormant_plan(state, trajectory, &plan)) return false;
  for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
       ++slot) {
   const Record& record = state->records[slot];
   if (record.matter_q8 != 0u && record.lane[1] == owner &&
       record.lane[0] == causal_rewrite::kFormTrajectoryTerm &&
       (record.lane[3] != 0u || record.lane[6] != 0u ||
        record.lane[7] != 0u || record.reserved[0] != 0u ||
        record.reserved[1] != 0u))
    return false;
   if (record.matter_q8 != 0u && record.lane[1] == owner &&
       record.lane[0] == causal_rewrite::kFormTrajectoryProvenance)
    return false;
  }
  return true;
 }
 const std::uint32_t blocks = (trajectory.lane[2] + 1u) / 2u;
 std::uint32_t rolling = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
  std::uint32_t term_matches = 0u;
  std::uint32_t provenance_matches = 0u;
  for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
       ++slot) {
   const Record& record = state->records[slot];
   if (record.matter_q8 == 0u || record.lane[1] != owner ||
       record.lane[2] != ordinal)
    continue;
   if (record.lane[0] == causal_rewrite::kFormTrajectoryTerm) {
    if (record.lane[3] != 0u || record.lane[6] != 0u ||
        record.lane[7] != 0u || record.reserved[0] != 0u ||
        record.reserved[1] != 0u)
     return false;
    ++term_matches;
   }
   provenance_matches +=
       record.lane[0] == causal_rewrite::kFormTrajectoryProvenance;
  }
  if (term_matches != 1u || provenance_matches > 1u) return false;
 }
 bool saw_provenance = false;
 bool saw_missing_provenance = false;
 for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
  const std::uint32_t provenance = causal_rewrite::find_owned_block(
      state, causal_rewrite::kFormTrajectoryProvenance, owner, ordinal);
  saw_provenance |= provenance != causal_rewrite::kInvalid;
  saw_missing_provenance |= provenance == causal_rewrite::kInvalid;
 }
 // Retained bound prefixes deliberately shed source provenance after END.
 // Mixed zero/nonzero provenance is malformed; accept either the declared
 // carried zero-provenance phenotype or a complete canonical set.
 if (saw_provenance && saw_missing_provenance) return false;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  const Record& record = state->records[slot];
  if (record.matter_q8 == 0u || record.lane[1] != owner ||
      record.lane[2] < blocks)
   continue;
  if (record.lane[0] == causal_rewrite::kFormTrajectoryTerm ||
      record.lane[0] == causal_rewrite::kFormTrajectoryProvenance)
   return false;
 }
 for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
  std::uint32_t word = 0u;
  if (!causal_rewrite::trajectory_word_at(state, owner, index, &word))
   return false;
  if (saw_provenance) {
   const std::uint32_t provenance_slot = causal_rewrite::find_owned_block(
       state, causal_rewrite::kFormTrajectoryProvenance, owner, index / 2u);
   if (provenance_slot == causal_rewrite::kInvalid) return false;
   const Record& provenance = state->records[provenance_slot];
   const std::uint32_t local = index % 2u;
   const std::uint32_t validity = 1u << local;
   const std::uint32_t origin = provenance.lane[3u + local * 2u];
   const std::uint32_t producer = provenance.lane[4u + local * 2u];
   if ((provenance.lane[7] & ~0x3u) != 0u ||
       (provenance.lane[7] & validity) == 0u || origin > 1u ||
       (origin == 0u && producer != causal_rewrite::kInvalid) ||
       (origin == 1u &&
        (producer == causal_rewrite::kInvalid ||
         producer >= causal_rewrite::live_record_capacity(state))))
    return false;
  }
  rolling = causal_rewrite::rewrite_mix(rolling, word, index);
 }
 return trajectory.lane[6] == rolling;
}

struct DormantRetentionEvidence {
 std::uint32_t slot = causal_rewrite::kInvalid;
 std::uint32_t support = 0u;
 std::uint32_t prefix = 0u;
 std::uint32_t age = 0u;
 std::uint32_t owner = causal_rewrite::kInvalid;
 std::uint32_t records = 0u;
};

BCC32_CROSS_CONTACT_HD inline std::uint32_t dormant_owned_record_count(
 const ResidentRewriteState* state, std::uint32_t owner) {
 if (state == nullptr || owner == causal_rewrite::kInvalid) return 0u;
 std::uint32_t count = 0u;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  const Record& record = state->records[slot];
  if (record.matter_q8 == 0u || record.lane[1] != owner) continue;
  if (record.lane[0] == causal_rewrite::kFormTrajectory ||
      record.lane[0] == causal_rewrite::kFormTrajectoryTerm ||
      record.lane[0] == causal_rewrite::kFormTrajectoryProvenance)
   ++count;
 }
 return count;
}

BCC32_CROSS_CONTACT_OUTLINE bool read_dormant_retention_evidence(
 ResidentRewriteState* state, std::uint32_t slot,
 DormantRetentionEvidence* evidence) {
 if (state == nullptr || evidence == nullptr ||
 !dormant_carry_identity_valid(state, slot))
 return false;
 const Record& trajectory = state->records[slot];
 evidence->slot = slot;
 evidence->prefix = trajectory.lane[2];
 evidence->age = trajectory.lane[4];
 evidence->owner = trajectory.lane[1];
 evidence->records = dormant_owned_record_count(state, trajectory.lane[1]);
 // Missing or ambiguous live authority is valid evidence of zero retention
 // value; it must not become a new selection or continuation route.
 SpanPrefixWitness witness{};
 if (is_compact_dormant_history(trajectory)) {
  std::uint32_t lengths[causal_rewrite::kMaximumProgramVariables]{};
  if (compact_program_and_lengths(
          state, trajectory, &witness.program_slot, lengths))
   witness.found = true;
 } else {
  (void)find_unique_bound_span_prefix_impl(
      state, slot, &witness, true);
 }
 if (witness.found &&
     witness.program_slot < causal_rewrite::live_record_capacity(state)) {
 const Record& program = state->records[witness.program_slot];
 if (program.matter_q8 != 0u &&
 program.lane[0] == causal_rewrite::kFormSpanProgram &&
 causal_rewrite::resident_program_authoritative(
 state, witness.program_slot))
 evidence->support = program.lane[3];
 }
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool dormant_retention_is_worse(
 const ResidentRewriteState* state,
 const DormantRetentionEvidence& candidate,
 const DormantRetentionEvidence& incumbent) {
 if (candidate.support != incumbent.support)
 return candidate.support < incumbent.support;
 // Once learned authority is equal, prefer the history that consumes less of
 // the one shared Record ecology. This is a resource law, not a semantic
 // preference: no content, token, or actor label enters the score. Dormancy
 // age is physical time in the resident epoch and is used only after support,
 // footprint, and matched-prefix extent agree.
 if (candidate.records != incumbent.records)
 return candidate.records > incumbent.records;
 if (candidate.prefix != incumbent.prefix)
 return candidate.prefix < incumbent.prefix;
 if (candidate.age != incumbent.age)
 return candidate.age > incumbent.age;
 // Physical owners and slots are allocation artifacts. Resolve an otherwise
 // equal retention rank by exact trajectory content so record permutation
 // cannot choose different discourse. Exact duplicates are interchangeable.
 const std::uint32_t shared = candidate.prefix < incumbent.prefix
                                  ? candidate.prefix
                                  : incumbent.prefix;
 CompactDormantPlan candidate_plan{};
 CompactDormantPlan incumbent_plan{};
 const bool candidate_compact =
     is_compact_dormant_history(state->records[candidate.slot]);
 const bool incumbent_compact =
     is_compact_dormant_history(state->records[incumbent.slot]);
 if ((candidate_compact &&
      !build_compact_dormant_plan(
          state, state->records[candidate.slot], &candidate_plan)) ||
     (incumbent_compact &&
      !build_compact_dormant_plan(
          state, state->records[incumbent.slot], &incumbent_plan)))
  return false;
 for (std::uint32_t index = 0u; index < shared; ++index) {
  std::uint32_t candidate_word = 0u;
  std::uint32_t incumbent_word = 0u;
  const bool candidate_ok = candidate_compact
      ? compact_word_from_plan(
            state, state->records[candidate.slot], candidate_plan, index,
            &candidate_word)
      : causal_rewrite::trajectory_word_at(
            state, state->records[candidate.slot].lane[1], index,
            &candidate_word);
  const bool incumbent_ok = incumbent_compact
      ? compact_word_from_plan(
            state, state->records[incumbent.slot], incumbent_plan, index,
            &incumbent_word)
      : causal_rewrite::trajectory_word_at(
            state, state->records[incumbent.slot].lane[1], index,
            &incumbent_word);
  if (!candidate_ok || !incumbent_ok)
   return false;
  if (candidate_word != incumbent_word)
   return candidate_word < incumbent_word;
 }
 return false;
}

BCC32_CROSS_CONTACT_HD inline void clear_trajectory_with_provenance(
 ResidentRewriteState* state, std::uint32_t slot);

BCC32_CROSS_CONTACT_HD inline bool reclaim_one_dormant_for_pressure(
 ResidentRewriteState* state) {
 if (state == nullptr) return false;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  if (is_dormant_history(state->records[slot]) &&
      !dormant_carry_identity_valid(state, slot)) {
   state->fault = 1u;
   return false;
  }
 }
 DormantRetentionEvidence worst{};
 bool have_worst = false;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  if (!is_dormant_history(state->records[slot])) continue;
  DormantRetentionEvidence candidate{};
  if (!read_dormant_retention_evidence(state, slot, &candidate)) {
   state->fault = 1u;
   return false;
  }
  if (!have_worst || dormant_retention_is_worse(state, candidate, worst)) {
   worst = candidate;
   have_worst = true;
  }
 }
 if (!have_worst) return false;
 clear_trajectory_with_provenance(state, worst.slot);
 ++state->revision;
 return true;
}

BCC32_CROSS_CONTACT_OUTLINE void reclaim_dormant_for_record_pressure(
 ResidentRewriteState* state, std::uint32_t desired_free) {
 if (state == nullptr || state->fault != 0u) return;
 while (causal_rewrite::free_record_count(state) < desired_free) {
  if (!reclaim_one_dormant_for_pressure(state)) break;
  if (state->fault != 0u) return;
 }
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t first_dormant_carry(
 const ResidentRewriteState* state) {
 if (state == nullptr) return causal_rewrite::kInvalid;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot)
 if (is_dormant_history(state->records[slot])) return slot;
 return causal_rewrite::kInvalid;
}

BCC32_CROSS_CONTACT_HD inline void clear_trajectory_with_provenance(
 ResidentRewriteState* state, std::uint32_t slot) {
 if (state == nullptr || slot >= causal_rewrite::live_record_capacity(state)) return;
 const Record& trajectory = state->records[slot];
 if (trajectory.matter_q8 == 0u ||
 trajectory.lane[0] != causal_rewrite::kFormTrajectory)
 return;
 const std::uint32_t owner = trajectory.lane[1];
 causal_rewrite::clear_owned_records(
 state, causal_rewrite::kFormTrajectoryProvenance, owner);
 causal_rewrite::clear_trajectory(state, slot);
}

BCC32_CROSS_CONTACT_HD inline bool detach_active_carry_before_new_contact(
 ResidentRewriteState* state) {
 if (state == nullptr) return false;
 const std::uint32_t slot = causal_rewrite::find_current_trajectory(state);
 if (slot == causal_rewrite::kInvalid) return false;
 Record& trajectory = state->records[slot];
 if (!is_carried_history(trajectory) ||
 (trajectory.lane[7] & causal_rewrite::kTrajectoryHasGenerated) != 0u)
 return false;
 trajectory.lane[3] = 1u;
 trajectory.lane[4] = 0u;
 ++trajectory.revision;
 ++state->revision;
 const std::uint32_t owner = trajectory.lane[1];
 const bool compacted = compact_dormant_history(state, slot);
 if (!compacted)
  causal_rewrite::clear_owned_records(
      state, causal_rewrite::kFormTrajectoryProvenance, owner);
 reclaim_dormant_for_record_pressure(state, kDormantOperationalReserve);
 if (state->fault != 0u ||
     causal_rewrite::free_record_count(state) < kDormantOperationalReserve)
  return false;
 // The next physical word belongs to a genuinely new contact. Demoting the
 // Carry alone leaves no current trajectory, so seed the ordinary empty
 // current owner before the new word is admitted. The dormant prefix remains
 // physically separate until PAUSE decides whether it may rejoin.
 return causal_rewrite::ensure_current_trajectory(state) !=
        causal_rewrite::kInvalid;
}

BCC32_CROSS_CONTACT_HD inline bool trajectory_has_provenance(
 const ResidentRewriteState* state, const Record& trajectory) {
 if (state == nullptr) return false;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot) {
 const Record& record = state->records[slot];
 if (record.matter_q8 != 0u &&
 record.lane[0] == causal_rewrite::kFormTrajectoryProvenance &&
 record.lane[1] == trajectory.lane[1])
 return true;
 }
 return false;
}

// This validates complete physical provenance without granting generated
// events external authority. Compact metadata records that distinction below.
BCC32_CROSS_CONTACT_HD inline bool trajectory_has_complete_provenance(
 const ResidentRewriteState* state, const Record& trajectory) {
 if (state == nullptr || trajectory.lane[2] == 0u) return false;
 for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
  const std::uint32_t block = causal_rewrite::find_owned_block(
      state, causal_rewrite::kFormTrajectoryProvenance, trajectory.lane[1],
      index / 2u);
  if (block == causal_rewrite::kInvalid) return false;
  const Record& provenance = state->records[block];
  const std::uint32_t local = index % 2u;
  const std::uint32_t origin = provenance.lane[3u + local * 2u];
  const std::uint32_t producer = provenance.lane[4u + local * 2u];
  if (provenance.matter_q8 == 0u ||
      provenance.lane[0] != causal_rewrite::kFormTrajectoryProvenance ||
      (provenance.lane[causal_rewrite::kProvenanceValidityLane] & ~0x3u) != 0u ||
      (provenance.lane[causal_rewrite::kProvenanceValidityLane] &
       (1u << local)) == 0u ||
      (origin != causal_rewrite::kProvenanceExternalOrigin &&
       origin != causal_rewrite::kProvenanceGeneratedOrigin) ||
      (origin == causal_rewrite::kProvenanceExternalOrigin &&
       producer != causal_rewrite::kInvalid) ||
      (origin == causal_rewrite::kProvenanceGeneratedOrigin &&
       (producer == causal_rewrite::kInvalid ||
        producer >= causal_rewrite::live_record_capacity(state))))
   return false;
 }
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool trajectory_is_completely_external(
 const ResidentRewriteState* state, const Record& trajectory) {
 if (state == nullptr || trajectory.lane[2] == 0u) return false;
 for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
 const std::uint32_t block = causal_rewrite::find_owned_block(
 state, causal_rewrite::kFormTrajectoryProvenance,
 trajectory.lane[1], index / 2u);
 if (block == causal_rewrite::kInvalid) return false;
 const Record& provenance = state->records[block];
 const std::uint32_t local = index % 2u;
 const std::uint32_t valid = 1u << local;
 const std::uint32_t origin_lane = 3u + local * 2u;
 const std::uint32_t producer_lane = 4u + local * 2u;
 if ((provenance.lane[causal_rewrite::kProvenanceValidityLane] &
 ~0x3u) != 0u ||
 (provenance.lane[causal_rewrite::kProvenanceValidityLane] &
 valid) == 0u ||
 provenance.lane[origin_lane] !=
 causal_rewrite::kProvenanceExternalOrigin ||
 provenance.lane[producer_lane] != causal_rewrite::kInvalid)
 return false;
 }
 return true;
}

struct JoinProbeEscrow {
 std::uint32_t owner = causal_rewrite::kInvalid;
 std::uint32_t allocation_cursor = 0u;
 std::uint32_t fault = 0u;
};

BCC32_CROSS_CONTACT_HD inline void rollback_probe_records(
 ResidentRewriteState* state, const JoinProbeEscrow& escrow) {
 if (state == nullptr || escrow.owner == causal_rewrite::kInvalid) return;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
      ++slot) {
  Record& record = state->records[slot];
  if (record.matter_q8 == 0u || record.lane[1] != escrow.owner ||
      (record.lane[0] != causal_rewrite::kFormTrajectory &&
       record.lane[0] != causal_rewrite::kFormTrajectoryTerm))
   continue;
  const std::uint32_t matter = record.matter_q8;
  const std::uint32_t revision = record.revision - 1u;
  record = Record{};
  record.lane[0] = causal_rewrite::kFormEmpty;
  record.matter_q8 = matter;
  record.revision = revision;
 }
 state->allocation_cursor = escrow.allocation_cursor;
 state->fault = escrow.fault;
}

BCC32_CROSS_CONTACT_HD inline bool joined_word_at(
 const ResidentRewriteState* state, const Record& dormant,
 const Record& current, std::uint32_t index, std::uint32_t* word) {
 if (word == nullptr) return false;
 if (index < dormant.lane[2])
 return compact_dormant_word_at(state, dormant, index, word);
 const std::uint32_t current_index = index - dormant.lane[2];
 return current_index < current.lane[2] &&
 causal_rewrite::trajectory_word_at(
  state, current.lane[1], current_index, word);
}

BCC32_CROSS_CONTACT_HD inline bool current_owned_reclaimable_record(
 const Record& record, std::uint32_t current_slot,
 std::uint32_t candidate_slot, std::uint32_t current_owner) {
 if (candidate_slot == current_slot) return true;
 return record.matter_q8 != 0u && record.lane[1] == current_owner &&
        (record.lane[0] == causal_rewrite::kFormTrajectoryTerm ||
         record.lane[0] == causal_rewrite::kFormTrajectoryProvenance);
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t current_owned_reclaimable_records(
 const ResidentRewriteState* state, const Record& current) {
 if (state == nullptr) return 0u;
 std::uint32_t count = 1u;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot) {
 const Record& record = state->records[slot];
 if (&record == &current || record.matter_q8 == 0u ||
 record.lane[1] != current.lane[1])
 continue;
 if (record.lane[0] == causal_rewrite::kFormTrajectoryTerm ||
 record.lane[0] == causal_rewrite::kFormTrajectoryProvenance)
 ++count;
 }
 return count;
}

// Probe with temporary ordinary Trajectory Records so the existing canonical
// SpanProgram matcher remains the only interpreter. Every byte touched by a
// failed probe, including the allocation cursor and fault word, is restored.
BCC32_CROSS_CONTACT_OUTLINE bool probe_dormant_join(
 ResidentRewriteState* state, std::uint32_t dormant_slot,
 std::uint32_t current_slot,
 causal_rewrite::ProgramCandidateConsensus* result) {
 if (state == nullptr || result == nullptr ||
 dormant_slot >= causal_rewrite::live_record_capacity(state) ||
 current_slot >= causal_rewrite::live_record_capacity(state))
 return false;
 const Record& dormant = state->records[dormant_slot];
 const Record& current = state->records[current_slot];
 if (!is_dormant_history(dormant) ||
 current.matter_q8 == 0u ||
 current.lane[0] != causal_rewrite::kFormTrajectory ||
 current.lane[3] != 0u || current.lane[7] != 0u ||
 current.lane[2] == 0u)
 return false;

 if (dormant.lane[2] > causal_rewrite::kMaximumTrajectoryEvents ||
 current.lane[2] >
 causal_rewrite::kMaximumTrajectoryEvents - dormant.lane[2])
 return false;
 const std::uint32_t extent = dormant.lane[2] + current.lane[2];
 const std::uint32_t blocks = (extent + 1u) / 2u;
 const std::uint32_t required = blocks + 1u;
 if (causal_rewrite::free_record_count(state) < required)
 return false;

 // Read both histories before borrowing any current-contact Record. The probe
 // may then reuse those slots without losing its own source bytes.
 std::uint32_t joined_words[causal_rewrite::kMaximumTrajectoryEvents]{};
 CompactDormantPlan dormant_plan{};
 const bool dormant_compact = is_compact_dormant_history(dormant);
 if (dormant_compact &&
     !build_compact_dormant_plan(state, dormant, &dormant_plan))
  return false;
 if (dormant_compact) {
  std::uint32_t source = causal_rewrite::kInvalid;
  std::uint32_t lengths[causal_rewrite::kMaximumProgramVariables]{};
  if (!compact_program_and_lengths(state, dormant, &source, lengths))
   return false;
 }
 for (std::uint32_t index = 0u; index < extent; ++index) {
  const bool ok = index < dormant.lane[2]
      ? (dormant_compact
             ? compact_word_from_plan(
                   state, dormant, dormant_plan, index, &joined_words[index])
             : causal_rewrite::trajectory_word_at(
                   state, dormant.lane[1], index, &joined_words[index]))
      : causal_rewrite::trajectory_word_at(
            state, current.lane[1], index - dormant.lane[2],
            &joined_words[index]);
  if (!ok) return false;
 }

 JoinProbeEscrow escrow{};
 escrow.allocation_cursor = state->allocation_cursor;
 escrow.fault = state->fault;
 const std::uint32_t owner = causal_rewrite::make_record_owner(
 state, causal_rewrite::rewrite_mix(
 dormant.lane[1], current.lane[1], extent ^ 0x23d1501u));
 if (owner == causal_rewrite::kInvalid) return false;
 escrow.owner = owner;

 // The reserve preflight above guarantees these allocations. Every borrowed
 // slot was a canonical Empty Record; the collision-free owner makes the
 // complete temporary set discoverable for byte-exact rollback without a
 // second stack-resident Record copy.
 const std::uint32_t header_slot = causal_rewrite::allocate_record(state);
 if (header_slot == causal_rewrite::kInvalid) {
  rollback_probe_records(state, escrow);
  return false;
 }
 causal_rewrite::clear_record(&state->records[header_slot]);
 Record& header = state->records[header_slot];
 header.lane[0] = causal_rewrite::kFormTrajectory;
 header.lane[1] = owner;
 header.lane[2] = extent;
 header.lane[3] = 0u;
 header.lane[4] = 0u;
 header.lane[5] = causal_rewrite::kInvalid;
 header.lane[6] = 0u;
 header.lane[7] = 0u;

 for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
 const std::uint32_t term_slot = causal_rewrite::allocate_record(state);
 if (term_slot == causal_rewrite::kInvalid) {
  rollback_probe_records(state, escrow);
  return false;
 }
 causal_rewrite::clear_record(&state->records[term_slot]);
 Record& term = state->records[term_slot];
 term.lane[0] = causal_rewrite::kFormTrajectoryTerm;
 term.lane[1] = owner;
 term.lane[2] = ordinal;
 const std::uint32_t first = ordinal * 2u;
 term.lane[4] = joined_words[first];
 if (first + 1u < extent) term.lane[5] = joined_words[first + 1u];
 header.lane[6] = causal_rewrite::rewrite_mix(
     header.lane[6], joined_words[first], first);
 if (first + 1u < extent)
  header.lane[6] = causal_rewrite::rewrite_mix(
      header.lane[6], joined_words[first + 1u], first + 1u);
 }

 causal_rewrite::ProgramCandidateConsensus consensus{};
 causal_rewrite::collect_resident_span_program_candidates(
 state, header, &consensus);
  *result = consensus;
 rollback_probe_records(state, escrow);
 return true;
}

struct DormantContinuationPlan {
 std::uint32_t dormant_slot = causal_rewrite::kInvalid;
 std::uint32_t extent = 0u;
 std::uint32_t support = 0u;
 bool found = false;
 bool tied = false;
};

// RWR25 does not introduce a referent table, entity id, recency winner, or
// dialogue score. A dormant history can compete only through evidence already
// owned by the authoritative SpanProgram that would continue it. Structural
// Resident support is primary. Equal support across distinct compatible
// histories remains unresolved even when one retained prefix is longer: raw
// extent is not evidence of which history the living Program should continue.
//
// Keeping this merge independent of Record slot is important for two reasons:
// * the actual selector below can be tested under arbitrary traversal order;
// * later counterevidence/reafference may change SpanProgram support without
// teaching this layer any semantic notion of correction or identity.
struct DormantContinuationEvidence {
 std::uint32_t dormant_slot = causal_rewrite::kInvalid;
 std::uint32_t extent = 0u;
 std::uint32_t support = 0u;
 bool valid = false;
};

BCC32_CROSS_CONTACT_HD inline void merge_dormant_continuation_evidence(
 DormantContinuationPlan* best,
 const DormantContinuationEvidence& candidate) {
 if (best == nullptr || !candidate.valid) return;
 if (!best->found || candidate.support > best->support) {
 best->dormant_slot = candidate.dormant_slot;
 best->extent = candidate.extent;
 best->support = candidate.support;
 best->found = true;
 best->tied = false;
 return;
 }
 if (candidate.support == best->support &&
     candidate.dormant_slot != best->dormant_slot)
 best->tied = true;
}

BCC32_CROSS_CONTACT_HD inline DormantContinuationPlan
select_dormant_continuation(ResidentRewriteState* state,
 std::uint32_t current_slot) {
 DormantContinuationPlan best{};
 if (state == nullptr) return best;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot) {
 if (!is_dormant_history(state->records[slot])) continue;
 causal_rewrite::ProgramCandidateConsensus consensus{};
 if (!probe_dormant_join(state, slot, current_slot, &consensus) ||
 !consensus.have_candidate || consensus.conflict ||
 consensus.span_saw_unbound || consensus.span_saw_ambiguous)
 continue;
 // probe_dormant_join invokes only collect_resident_span_program_candidates.
 // Do not allow a future non-span extension of that collector to silently gain
 // reference authority here: this seam is explicitly paid by mature resident
 // SpanProgram evidence.
 if (!consensus.selected_from_span ||
 consensus.selected_support < causal_rewrite::kSpanProgramMatureSupport)
 continue;
 const std::uint32_t extent =
 state->records[slot].lane[2] +
 state->records[current_slot].lane[2];
 merge_dormant_continuation_evidence(
 &best, DormantContinuationEvidence{
 slot, extent, consensus.selected_support, true});
 }
 return best;
}

BCC32_CROSS_CONTACT_OUTLINE bool materialize_dormant_join(
 ResidentRewriteState* state, std::uint32_t dormant_slot,
 std::uint32_t current_slot) {
 if (state == nullptr ||
 dormant_slot >= causal_rewrite::live_record_capacity(state) ||
 current_slot >= causal_rewrite::live_record_capacity(state))
 return false;
 const Record dormant = state->records[dormant_slot];
 const Record current = state->records[current_slot];
 if (!is_dormant_history(dormant) ||
 current.matter_q8 == 0u ||
 current.lane[0] != causal_rewrite::kFormTrajectory ||
 current.lane[3] != 0u || current.lane[7] != 0u ||
 current.lane[2] == 0u)
 return false;

 if (dormant.lane[2] > causal_rewrite::kMaximumTrajectoryEvents ||
 current.lane[2] >
 causal_rewrite::kMaximumTrajectoryEvents - dormant.lane[2])
 return false;
 const std::uint32_t extent = dormant.lane[2] + current.lane[2];
 const bool dormant_compact = is_compact_dormant_history(dormant);
 CompactDormantPlan dormant_plan{};
 if (dormant_compact &&
     !build_compact_dormant_plan(state, dormant, &dormant_plan))
  return false;
 const bool dormant_has_provenance =
 trajectory_has_provenance(state, dormant);
 const bool current_has_provenance =
 trajectory_has_provenance(state, current);
 const bool dormant_external = dormant_compact
     ? dormant_plan.external_provenance
     : trajectory_is_completely_external(state, dormant);
 const bool current_external =
 trajectory_is_completely_external(state, current);
 if ((dormant_has_provenance && !dormant_external) ||
     (current_has_provenance && !current_external))
  return false;
 // A compact wholly-external fact is not authority by itself. It restores
 // ordinary external stamps only when the joining contact independently proves
 // wholly-external provenance. Mixed/generated and corrupt compact carry can
 // still be context, but can never launder evidence.
 const bool preserve_provenance = dormant_external && current_external;
 // Read every source byte before changing one Record. After this point the
 // materialization path is a bounded, failure-free transaction.
 std::uint32_t joined_words[causal_rewrite::kMaximumTrajectoryEvents]{};
 for (std::uint32_t index = 0u; index < extent; ++index) {
 if (index < dormant.lane[2]) {
  const bool ok = dormant_compact
      ? compact_word_from_plan(
            state, dormant, dormant_plan, index, &joined_words[index])
      : causal_rewrite::trajectory_word_at(
            state, dormant.lane[1], index, &joined_words[index]);
  if (!ok)
   return false;
  continue;
 }
 if (!causal_rewrite::trajectory_word_at(
 state, current.lane[1], index - dormant.lane[2],
 &joined_words[index]))
 return false;
 }

 const std::uint32_t blocks = (extent + 1u) / 2u;
 const std::uint32_t required =
 1u + blocks + (preserve_provenance ? blocks : 0u);
 if (causal_rewrite::free_record_count(state) +
 current_owned_reclaimable_records(state, state->records[current_slot]) <
 required)
 return false;

 const std::uint32_t new_owner = causal_rewrite::make_record_owner(
 state, causal_rewrite::rewrite_mix(
 dormant.lane[1], current.lane[1], extent ^ 0x23d1502u));
 if (new_owner == causal_rewrite::kInvalid) return false;

 // Plan the exact slots the ordinary allocator would encounter after the
 // current trajectory is reclaimed. This makes the capacity preflight
 // transactional: no allocation can fail after the first mutation.
 std::uint32_t planned_slots[
 1u + causal_rewrite::kMaximumTrajectoryEvents]{};
 std::uint32_t planned_count = 0u;
 std::uint32_t planned_cursor = state->allocation_cursor;
 for (std::uint32_t offset = 0u;
 offset < causal_rewrite::live_record_capacity(state) && planned_count < required;
 ++offset) {
 const std::uint32_t slot =
 (state->allocation_cursor + offset) % causal_rewrite::live_record_capacity(state);
 const Record& record = state->records[slot];
 const bool already_empty = record.matter_q8 != 0u &&
 record.lane[0] == causal_rewrite::kFormEmpty;
 if (!already_empty && !current_owned_reclaimable_record(
 record, current_slot, slot, current.lane[1]))
 continue;
 planned_slots[planned_count++] = slot;
 planned_cursor = (slot + 1u) % causal_rewrite::live_record_capacity(state);
 }
 if (planned_count != required) return false;

 clear_trajectory_with_provenance(state, current_slot);
 state->allocation_cursor = planned_cursor;
 std::uint32_t next_planned = 0u;
 const std::uint32_t header_slot = planned_slots[next_planned++];
 Record& joined = state->records[header_slot];
 joined.lane[0] = causal_rewrite::kFormTrajectory;
 joined.lane[1] = new_owner;
 joined.lane[2] = extent;
 joined.lane[3] = 0u;
 joined.lane[4] = 0u;
 joined.lane[5] = causal_rewrite::kInvalid;
 joined.lane[6] = 0u;
 joined.lane[7] = kTrajectoryHasCarry;
 ++joined.revision;

 for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
 const std::uint32_t term_slot = planned_slots[next_planned++];
 Record& term = state->records[term_slot];
 term.lane[0] = causal_rewrite::kFormTrajectoryTerm;
 term.lane[1] = new_owner;
 term.lane[2] = ordinal;
 ++term.revision;
 }

 for (std::uint32_t index = 0u; index < extent; ++index) {
 const std::uint32_t word = joined_words[index];
 const std::uint32_t term_slot = planned_slots[1u + index / 2u];
 state->records[term_slot].lane[4u + (index % 2u)] = word;
 ++state->records[term_slot].revision;
 joined.lane[6] =
 causal_rewrite::rewrite_mix(joined.lane[6], word, index);
 }

 if (preserve_provenance) {
 for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
 const std::uint32_t provenance_slot = planned_slots[next_planned++];
 Record& provenance = state->records[provenance_slot];
 provenance.lane[0] = causal_rewrite::kFormTrajectoryProvenance;
 provenance.lane[1] = new_owner;
 provenance.lane[2] = ordinal;
 provenance.lane[3] = causal_rewrite::kProvenanceExternalOrigin;
 provenance.lane[4] = causal_rewrite::kInvalid;
 provenance.lane[5] = causal_rewrite::kProvenanceExternalOrigin;
 provenance.lane[6] = causal_rewrite::kInvalid;
 provenance.lane[causal_rewrite::kProvenanceValidityLane] =
 ordinal + 1u == blocks && (extent & 1u) != 0u ? 1u : 3u;
 ++provenance.revision;
 }
 }

 // The selected dormant history has now been copied into the joined current
 // trajectory. Keeping both would duplicate context and can starve the next
 // generated term. Reclaim only after the replacement is complete.
 clear_trajectory_with_provenance(state, dormant_slot);
 ++state->revision;
 return true;
}

BCC32_CROSS_CONTACT_HD inline bool
prepare_dormant_continuation_before_pause(ResidentRewriteState* state) {
 if (state == nullptr || state->fault != 0u) return false;
 const std::uint32_t current =
 causal_rewrite::find_current_trajectory(state);
 if (current == causal_rewrite::kInvalid ||
 state->records[current].lane[7] != 0u ||
 state->records[current].lane[2] == 0u)
 return false;

 const DormantContinuationPlan plan =
 select_dormant_continuation(state, current);
 if (!plan.found) return false;
 if (plan.tied) {
 // This is deliberately the same fail-closed ambiguity surface consumed by
 // the continuing adult. RWR25 does not manufacture a question here. The
 // separately owned RWR24 clarification/resume mechanism may observe the
 // unresolved resident alternatives through its ordinary production path and
 // emit only a clarification surface it learned from external episodes.
 ++state->span_ambiguous_abstentions;
 return false;
 }
 return materialize_dormant_join(
 state, plan.dormant_slot, current);
}

BCC32_CROSS_CONTACT_OUTLINE void age_dormant_histories_at_end(
 ResidentRewriteState* state) {
 if (state == nullptr) return;
 bool changed = false;
 for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
 ++slot) {
 Record& trajectory = state->records[slot];
 if (!is_dormant_history(trajectory)) continue;
 if (trajectory.lane[4] != 0xffffffffu) ++trajectory.lane[4];
 ++trajectory.revision;
 changed = true;
 }

 if (changed) ++state->revision;
}
