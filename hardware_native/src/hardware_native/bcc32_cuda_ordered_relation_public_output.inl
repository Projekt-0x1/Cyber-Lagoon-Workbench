#pragma once

// Included inside bcc32_cuda_adult_stream_v1 after QueryAnswerReceipt.  This
// file owns only the final resident-unit -> public-byte transaction; relation
// selection and qualification remain in ordered resident tissue.

constexpr std::uint32_t kQueryOutputProducerNone = 0u;
constexpr std::uint32_t kQueryOutputProducerOrderedRelation = 1u;
constexpr std::uint32_t kQueryOutputProducerRelationClarification = 2u;

__global__ void latch_ordered_relation_execution_receipt_kernel(
    adult::ordered_relation::tissue::TissueView resident_tissue,
    adult::ordered_relation::OrderedRelationExecutionReceipt* execution,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || execution == nullptr ||
      receipt == nullptr)
    return;
  receipt->ordered_relation_route_authorized = execution->route_authorized;
  receipt->ordered_relation_topology_matches = execution->topology_matches;
  receipt->ordered_relation_qualified_candidates =
      execution->qualified_candidates;
  receipt->ordered_relation_withdrawn_candidates =
      execution->withdrawn_candidates;
  receipt->ordered_relation_invalid_sources = execution->invalid_sources;
  receipt->ordered_relation_conflict = execution->conflict;
  receipt->ordered_relation_ready = execution->ready;
  receipt->ordered_relation_clarification_ready =
      execution->clarification_ready;
  const bool source_current = execution->route_authorized != 0u &&
      execution->topology_matches != 0u &&
      adult::ordered_relation::ordered_relation_execution_receipt_is_current(
          resident_tissue, *execution);
  // Freeze producer authority inside the private execution transaction before
  // this contact can revise tissue. QueryAnswerReceipt only mirrors the result.
  execution->public_source_validated = source_current ? 1u : 0u;
  receipt->ordered_relation_source_current = source_current ? 1u : 0u;
  receipt->ordered_relation_stale_source_rejected =
      execution->route_authorized != 0u &&
          execution->topology_matches != 0u && !source_current
      ? 1u
      : 0u;
  receipt->ordered_relation_source_binding = execution->source_binding_index;
  receipt->ordered_relation_composed_source_binding =
      execution->composed_source_binding_index;
  receipt->ordered_relation_terminal_source_binding =
      execution->terminal_source_binding_index;
  receipt->ordered_relation_output_units = execution->output_unit_count;
  receipt->ordered_relation_composition_depth = execution->composition_depth;
  receipt->ordered_relation_tissue_revision = execution->tissue_revision;
  receipt->ordered_relation_source_revision =
      execution->source_evidence_revision;
  receipt->ordered_relation_composed_source_revision =
      execution->composed_source_evidence_revision;
  receipt->ordered_relation_terminal_source_revision =
      execution->terminal_source_evidence_revision;
  receipt->ordered_relation_ticketed_return_revision =
      execution->ticketed_return_evidence_revision;
  receipt->ordered_relation_ticketed_return_source_mask =
      execution->ticketed_return_source_mask;
  receipt->ordered_relation_source_positive_mass =
      execution->source_positive_mass;
  receipt->ordered_relation_source_counterevidence =
      execution->source_counterevidence;
  // Once exact resident topology matched, this route owns either its unique
  // output or its abstention. Marking an attempt suppresses legacy generation;
  // it does not make an ambiguous or withdrawn source emission-ready.
  if (execution->route_authorized != 0u &&
      execution->topology_matches != 0u)
    receipt->attempted = 1u;
}

__global__ void serialize_ordered_relation_public_output_kernel(
    const adult::ordered_relation::OrderedRelationExecutionReceipt* execution,
    const std::uint32_t* selected_units,
    const std::uint32_t* selected_unit_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit_count,
    const std::uint32_t* boundary_mask, std::uint8_t* output,
    std::uint32_t output_capacity, std::uint32_t* generated_count,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || execution == nullptr ||
      selected_unit_count == nullptr || generated_count == nullptr ||
      receipt == nullptr)
    return;
  if (execution->route_authorized == 0u ||
      execution->topology_matches == 0u)
    return;

  // This resident route owns its matched cue. Clear any previously staged
  // bytes before deciding whether its unique source can cross the membrane.
  generated_count[0] = 0u;
  receipt->candidate_producer = kQueryOutputProducerNone;
  if (execution->public_source_validated == 0u) {
    receipt->serialized_units = 0u;
    return;
  }
  const bool answer_ready = execution->ready != 0u &&
      execution->conflict == 0u;
  const bool clarification_ready = execution->clarification_ready != 0u &&
      execution->conflict != 0u;
  if ((!answer_ready && !clarification_ready) ||
      selected_units == nullptr || unit_lengths == nullptr ||
      unit_content == nullptr || boundary_mask == nullptr || output == nullptr ||
      selected_unit_count[0] == 0u ||
      selected_unit_count[0] != execution->output_unit_count)
    return;

  std::uint32_t required = 0u;
  std::uint32_t last_byte = 0x20u;
  for (std::uint32_t index = 0u; index < selected_unit_count[0]; ++index) {
    const std::uint32_t unit = selected_units[index];
    if (unit >= unit_count || unit_lengths[unit] == 0u)
      return;
    const std::uint32_t first = unit_content[unit * unit_words] & 0xffu;
    if (index != 0u && boundary_mask[last_byte] == 0u &&
        boundary_mask[first] == 0u)
      ++required;
    if (required > output_capacity - min(output_capacity, unit_lengths[unit]))
      return;
    required += unit_lengths[unit];
    const std::uint32_t final_offset = unit_lengths[unit] - 1u;
    const std::uint32_t final_word =
        unit_content[unit * unit_words + final_offset / 4u];
    last_byte = (final_word >> ((final_offset % 4u) * 8u)) & 0xffu;
  }
  if (required == 0u || required > output_capacity)
    return;

  std::uint32_t written = 0u;
  last_byte = 0x20u;
  for (std::uint32_t index = 0u; index < selected_unit_count[0]; ++index) {
    const std::uint32_t unit = selected_units[index];
    const std::uint32_t first = unit_content[unit * unit_words] & 0xffu;
    if (index != 0u && boundary_mask[last_byte] == 0u &&
        boundary_mask[first] == 0u)
      output[written++] = 0x20u;
    for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
      const std::uint32_t word =
          unit_content[unit * unit_words + offset / 4u];
      output[written++] =
          static_cast<std::uint8_t>(word >> ((offset % 4u) * 8u));
    }
    last_byte = output[written - 1u];
  }
  generated_count[0] = written;
  receipt->serialized_units = selected_unit_count[0];
  receipt->candidate_producer = clarification_ready
      ? kQueryOutputProducerRelationClarification
      : kQueryOutputProducerOrderedRelation;
}

// Surface and action realizers run after the direct relation candidate. They
// already preserve a nonzero generated_count, but an ambiguous, withdrawn, or
// malformed matched relation must remain silent even if another fallback later
// stages bytes. This final fence makes that abstention explicit and atomic.
__global__ void enforce_ordered_relation_public_silence_kernel(
    const adult::ordered_relation::OrderedRelationExecutionReceipt* execution,
    std::uint32_t* generated_count, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || execution == nullptr ||
      generated_count == nullptr || receipt == nullptr)
    return;
  if (execution->route_authorized == 0u ||
      execution->topology_matches == 0u)
    return;
  const bool source_current = execution->public_source_validated != 0u;
  const bool answer_owned = execution->ready != 0u &&
      execution->conflict == 0u &&
      source_current &&
      receipt->candidate_producer == kQueryOutputProducerOrderedRelation;
  const bool clarification_owned = execution->clarification_ready != 0u &&
      execution->conflict != 0u &&
      source_current &&
      receipt->candidate_producer ==
          kQueryOutputProducerRelationClarification;
  if (!answer_owned && !clarification_owned) {
    generated_count[0] = 0u;
    receipt->candidate_producer = kQueryOutputProducerNone;
    receipt->serialized_units = 0u;
  }
}
