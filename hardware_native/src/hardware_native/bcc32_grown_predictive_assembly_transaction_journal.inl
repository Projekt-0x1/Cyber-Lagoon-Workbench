BCC32_PA_DEVICE inline void set_error(DeviceState* state, std::uint32_t error) {
  state->error_bits |= error;
}
BCC32_PA_DEVICE inline bool journal_room(DeviceState* state) {
  if (state->journal_count < kJournalCapacity) return true;
  set_error(state, kErrorJournalFull);
  return false;
}
BCC32_PA_DEVICE inline bool push_journal(DeviceState* state, const JournalEntry& entry) {
  if (!journal_room(state)) return false;
  state->journal[state->journal_count++] = entry;
  return true;
}
BCC32_PA_DEVICE inline bool journal_cell(DeviceState* state, std::uint32_t population,
                                         std::uint32_t index) {
  JournalEntry entry{};
  entry.kind = JournalKind::kCell;
  entry.population = static_cast<std::uint8_t>(population);
  entry.index = static_cast<std::uint16_t>(index);
  entry.old_cell = state->populations[population].cells[index];
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_segment(DeviceState* state, std::uint32_t population,
                                            std::uint32_t index) {
  JournalEntry entry{};
  entry.kind = JournalKind::kSegment;
  entry.population = static_cast<std::uint8_t>(population);
  entry.index = static_cast<std::uint16_t>(index);
  entry.old_segment = state->populations[population].segments[index];
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_synapse(DeviceState* state, std::uint32_t population,
                                            std::uint32_t index) {
  JournalEntry entry{};
  entry.kind = JournalKind::kSynapse;
  entry.population = static_cast<std::uint8_t>(population);
  entry.index = static_cast<std::uint16_t>(index);
  entry.old_synapse = state->populations[population].synapses[index];
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_eligibility(DeviceState* state, std::uint32_t population,
                                                std::uint32_t index) {
  JournalEntry entry{};
  entry.kind = JournalKind::kEligibility;
  entry.population = static_cast<std::uint8_t>(population);
  entry.index = static_cast<std::uint16_t>(index);
  entry.old_eligibility = state->populations[population].eligibilities[index];
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_history(DeviceState* state, std::uint32_t index) {
  JournalEntry entry{};
  entry.kind = JournalKind::kHistory;
  entry.index = static_cast<std::uint16_t>(index);
  entry.old_history = state->history[index];
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_metadata(DeviceState* state) {
  if (state->metadata_snapshot_active != 0u) return true;
  JournalEntry entry{};
  entry.kind = JournalKind::kMetadata;
  entry.old_metadata.tick = state->tick;
  entry.old_metadata.predicted_mask = state->predicted_mask;
  entry.old_metadata.error_bits = state->error_bits;
  entry.old_metadata.transaction_mark = state->transaction_mark;
  entry.old_metadata.predicted_value = state->predicted_value;
  entry.old_metadata.prediction_active = state->prediction_active;
  entry.old_metadata.predicted_population = state->predicted_population;
  entry.old_metadata.predicted_segment = state->predicted_segment;
  entry.old_metadata.predicted_target = state->predicted_target;
  entry.old_metadata.publication_count = state->publication_count;
  entry.old_metadata.frozen_edge_class_mask = state->frozen_edge_class_mask;
  entry.old_metadata.represented_matter = state->represented_matter;
  entry.old_metadata.free_matter = state->free_matter;
  entry.old_metadata.internal_growth_calls = state->internal_growth_calls;
  entry.old_metadata.external_growth_calls = state->external_growth_calls;
  entry.old_metadata.canonical_snapshot_hash = state->canonical_snapshot_hash;
  entry.old_metadata.canonical_snapshot_tick = state->canonical_snapshot_tick;
  entry.old_metadata.canonical_snapshot_valid = state->canonical_snapshot_valid;
  state->transaction_contact_before = state->contact;
  for (std::uint32_t i = 0u; i < kPublicationCapacity; ++i)
    state->transaction_publications_before[i] = state->publications[i];
  state->metadata_snapshot_active = 1u;
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline bool journal_allocator(DeviceState* state, std::uint32_t population) {
  JournalEntry entry{};
  entry.kind = JournalKind::kAllocator;
  entry.population = static_cast<std::uint8_t>(population);
  const Population& pool = state->populations[population];
  entry.old_u32[0] = pool.free_segment_count;
  entry.old_u32[1] = pool.free_synapse_count;
  entry.old_u64[0] = pool.represented_matter;
  entry.old_u64[1] = pool.free_matter;
  return push_journal(state, entry);
}
BCC32_PA_DEVICE inline void rebuild_allocator(Population* pool) {
  pool->free_segment_count = 0u;
  pool->free_synapse_count = 0u;
  for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
    if (!pool->segments[i].live)
      pool->free_segments[pool->free_segment_count++] = static_cast<std::uint16_t>(i);
  }
  for (std::uint32_t i = 0u; i < kSynapsesPerPopulation; ++i) {
    if (!pool->synapses[i].live)
      pool->free_synapses[pool->free_synapse_count++] = static_cast<std::uint16_t>(i);
  }
}
struct DeviceState;
BCC32_PA_DEVICE inline std::uint64_t hash_state(const DeviceState& state);
BCC32_PA_DEVICE inline void begin_transaction(DeviceState* state) {
  state->transaction_mark = state->journal_count;
  state->metadata_snapshot_active = 0u;
}
BCC32_PA_DEVICE inline bool begin_canonical_snapshot(DeviceState* state) {
  if (state == nullptr) return false;
  state->canonical_snapshot_hash = hash_state(*state);
  state->canonical_snapshot_tick = state->tick;
  state->canonical_snapshot_valid = 1u;
  state->journal_count = 0u;
  state->transaction_mark = 0u;
  state->metadata_snapshot_active = 0u;
  return true;
}
BCC32_PA_DEVICE inline bool commit_contact_transaction(DeviceState* state) {
  if (state == nullptr || state->canonical_snapshot_valid == 0u) {
    if (state != nullptr) set_error(state, kErrorInvariant);
    return false;
  }
  state->journal_count = 0u;
  state->transaction_mark = 0u;
  state->metadata_snapshot_active = 0u;
  return true;
}
BCC32_PA_DEVICE inline bool freeze_route_class(DeviceState* state,
                                               std::uint32_t source_population,
                                               std::uint32_t target_population) {
  if (state == nullptr || !edge_allowed(source_population, target_population)) {
    if (state != nullptr) set_error(state, kErrorForbiddenEdge);
    return false;
  }
  state->frozen_edge_class_mask |=
      static_cast<std::uint16_t>(1u << route_class(source_population, target_population));
  return true;
}
BCC32_PA_DEVICE inline void inverse_to(DeviceState* state, std::uint32_t mark) {
  while (state->journal_count > mark) {
    const std::uint32_t journal_index = state->journal_count - 1u;
    const JournalEntry entry = state->journal[journal_index];
    switch (entry.kind) {
      case JournalKind::kCell:
        state->populations[entry.population].cells[entry.index] = entry.old_cell;
        break;
      case JournalKind::kSegment:
        state->populations[entry.population].segments[entry.index] = entry.old_segment;
        break;
      case JournalKind::kSynapse:
        state->populations[entry.population].synapses[entry.index] = entry.old_synapse;
        break;
      case JournalKind::kEligibility:
        state->populations[entry.population].eligibilities[entry.index] = entry.old_eligibility;
        break;
      case JournalKind::kHistory:
        state->history[entry.index] = entry.old_history;
        break;
      case JournalKind::kMetadata:
        state->tick = entry.old_metadata.tick;
        state->predicted_mask = entry.old_metadata.predicted_mask;
        state->error_bits = entry.old_metadata.error_bits;
        state->transaction_mark = entry.old_metadata.transaction_mark;
        state->predicted_value = entry.old_metadata.predicted_value;
        state->prediction_active = entry.old_metadata.prediction_active;
        state->predicted_population = entry.old_metadata.predicted_population;
        state->predicted_segment = entry.old_metadata.predicted_segment;
        state->predicted_target = entry.old_metadata.predicted_target;
        state->publication_count = entry.old_metadata.publication_count;
        state->frozen_edge_class_mask = entry.old_metadata.frozen_edge_class_mask;
        state->represented_matter = entry.old_metadata.represented_matter;
        state->free_matter = entry.old_metadata.free_matter;
        state->internal_growth_calls = entry.old_metadata.internal_growth_calls;
        state->external_growth_calls = entry.old_metadata.external_growth_calls;
        state->canonical_snapshot_hash = entry.old_metadata.canonical_snapshot_hash;
        state->canonical_snapshot_tick = entry.old_metadata.canonical_snapshot_tick;
        state->canonical_snapshot_valid = entry.old_metadata.canonical_snapshot_valid;
        state->contact = state->transaction_contact_before;
        for (std::uint32_t i = 0u; i < kPublicationCapacity; ++i)
          state->publications[i] = state->transaction_publications_before[i];
        state->metadata_snapshot_active = 0u;
        break;
      case JournalKind::kAllocator:
        state->populations[entry.population].free_segment_count = entry.old_u32[0];
        state->populations[entry.population].free_synapse_count = entry.old_u32[1];
        state->populations[entry.population].represented_matter = entry.old_u64[0];
        state->populations[entry.population].free_matter = entry.old_u64[1];
        break;
    }
    state->journal[journal_index] = JournalEntry{};
    --state->journal_count;
  }
  for (std::uint32_t population = 0u; population < kPopulationCount; ++population)
    rebuild_allocator(&state->populations[population]);
}
