inline void allocate_construction_organ(AdultState& state) {
  // The composer owns its role matter so it works with or without the
  // distributed surface organ (whose arrays only exist under
  // --distributed-motor).
  state.construction_role_projection.allocate(
      roles::role_projection_scratch_words(state.unit_capacity));
  state.construction_roles.allocate(state.unit_capacity);
  state.construction_role_population.allocate(roles::kStructuralRoleCount);
  state.construction_closure_bytes.allocate(4u);
  cuda_require(cudaMemset(state.construction_closure_bytes.get(), 0,
                          state.construction_closure_bytes.bytes()),
               "clear resident construction closure bytes");
  state.construction_closed_class_mask.allocate(state.unit_capacity);
  cuda_require(cudaMemset(state.construction_closed_class_mask.get(), 0,
                          state.construction_closed_class_mask.bytes()),
               "clear resident construction closed-class mask");
  state.construction_filler_terminal_mask.allocate(256u);
  cuda_require(cudaMemset(state.construction_filler_terminal_mask.get(), 0,
                          state.construction_filler_terminal_mask.bytes()),
               "clear resident filler terminal mask");
  state.construction_initial_form_mask.allocate(state.unit_capacity);
  cuda_require(cudaMemset(state.construction_initial_form_mask.get(), 0,
                          state.construction_initial_form_mask.bytes()),
               "clear resident initial-form mask");
  cuda_require(cudaMemset(state.construction_role_projection.get(), 0,
                          state.construction_role_projection.bytes()),
               "clear resident construction role projection");
  cuda_require(cudaMemset(state.construction_roles.get(), 0,
                          state.construction_roles.bytes()),
               "clear resident construction roles");
  cuda_require(cudaMemset(state.construction_role_population.get(), 0,
                          state.construction_role_population.bytes()),
               "clear resident construction role population");
  state.construction_tokens.allocate(
      static_cast<std::size_t>(construction::kConstructionCap) *
      construction::kConstructionMaxTokens);
  state.construction_lengths.allocate(construction::kConstructionCap);
  state.construction_slot_counts.allocate(construction::kConstructionCap);
  state.construction_supports.allocate(construction::kConstructionCap);
  const std::size_t construction_slot_count =
      static_cast<std::size_t>(construction::kConstructionCap) *
      construction::kConstructionMaxSlots;
  state.construction_slot_units.allocate(
      construction_slot_count * construction::kConstructionSlotPopulationCap);
  state.construction_slot_masses.allocate(
      construction_slot_count * construction::kConstructionSlotPopulationCap);
  state.construction_slot_totals.allocate(construction_slot_count);
  state.construction_slot_overflow.allocate(construction_slot_count);
  state.construction_hash_slots.allocate(construction::kConstructionHashCap);
  state.construction_store_count.allocate(1u);
  state.construction_evidence_revision.allocate(construction::kConstructionCap);
  state.construction_origin_revision.allocate(construction::kConstructionCap);
  state.construction_pool_units.allocate(construction::kConstructionPoolCap);
  state.construction_pool_roles.allocate(construction::kConstructionPoolCap);
  state.construction_pool_weights.allocate(construction::kConstructionPoolCap);
  state.construction_pool_meta.allocate(1u + roles::kStructuralRoleCount);
  state.construction_best.allocate(1u);
  state.construction_last_selected.allocate(1u);
  state.construction_plan.allocate(construction::kConstructionMaxTokens);
  state.construction_plan_meta.allocate(3u);
  cuda_require(cudaMemset(state.construction_lengths.get(), 0,
                          state.construction_lengths.bytes()),
               "clear resident construction lengths");
  cuda_require(cudaMemset(state.construction_slot_units.get(), 0xff,
                          state.construction_slot_units.bytes()),
               "clear resident construction slot units");
  cuda_require(cudaMemset(state.construction_evidence_revision.get(), 0,
                          state.construction_evidence_revision.bytes()),
               "clear resident construction evidence revisions");
  cuda_require(cudaMemset(state.construction_origin_revision.get(), 0,
                          state.construction_origin_revision.bytes()),
               "clear resident construction origin revisions");
  cuda_require(cudaMemset(state.construction_slot_masses.get(), 0,
                          state.construction_slot_masses.bytes()),
               "clear resident construction slot masses");
  cuda_require(cudaMemset(state.construction_slot_totals.get(), 0,
                          state.construction_slot_totals.bytes()),
               "clear resident construction slot totals");
  cuda_require(cudaMemset(state.construction_slot_overflow.get(), 0,
                          state.construction_slot_overflow.bytes()),
               "clear resident construction slot overflow");
  cuda_require(cudaMemset(state.construction_hash_slots.get(), 0,
                          state.construction_hash_slots.bytes()),
               "clear resident construction hash");
  cuda_require(cudaMemset(state.construction_store_count.get(), 0,
                          state.construction_store_count.bytes()),
               "clear resident construction extent");
  cuda_require(cudaMemset(state.construction_last_selected.get(), 0xff,
                          state.construction_last_selected.bytes()),
               "clear resident construction selection trace");
  cuda_require(cudaMemset(state.construction_plan_meta.get(), 0,
                          state.construction_plan_meta.bytes()),
               "clear resident construction plan");
  // Morphological agreement organ: learned suffix-class adjacency counts.
  // BCC32_LESION_MORPH_AGREEMENT nulls the signal (learning and gating are
  // skipped), which restores the exact pre-morphology composer behavior.
  state.construction_suffix_transitions.allocate(
      construction::kSuffixClassCount * construction::kSuffixClassCount);
  cuda_require(cudaMemset(state.construction_suffix_transitions.get(), 0,
                          state.construction_suffix_transitions.bytes()),
               "clear resident suffix-class adjacency counts");
  state.morph_agreement_lesioned =
      std::getenv("BCC32_LESION_MORPH_AGREEMENT") != nullptr;
  // Role-signal canonicalization matter. BCC32_LESION_ROLE_CANON restores
  // the exact legacy per-variant role derivation (nullptr canonical map).
  state.construction_role_canon.allocate(state.unit_capacity);
  state.construction_role_canon_signatures.allocate(state.unit_capacity);
  std::uint32_t canon_table_size = 1024u;
  while (canon_table_size < 2u * state.unit_capacity) canon_table_size <<= 1u;
  state.construction_role_canon_table_size = canon_table_size;
  state.construction_role_canon_keys.allocate(canon_table_size);
  state.construction_role_canon_reps.allocate(canon_table_size);
  cuda_require(cudaMemset(state.construction_role_canon.get(), 0,
                          state.construction_role_canon.bytes()),
               "clear resident role canonical map");
  state.role_canon_lesioned = std::getenv("BCC32_LESION_ROLE_CANON") != nullptr;
  state.entropy_glue_lesioned =
      std::getenv("BCC32_LESION_ENTROPY_GLUE") != nullptr;
  // Whole-reply content-commitment state (ordered plan + serialization
  // cursor). BCC32_LESION_CONTENT_COMMIT skips formation entirely, which
  // restores the exact pool-driven composer behavior.
  state.content_commitment_units.allocate(construction::kCommitmentCap);
  state.content_commitment_meta.allocate(2u);
  cuda_require(cudaMemset(state.content_commitment_units.get(), 0,
                          state.content_commitment_units.bytes()),
               "clear resident content commitment plan");
  cuda_require(cudaMemset(state.content_commitment_meta.get(), 0,
                          state.content_commitment_meta.bytes()),
               "clear resident content commitment cursor");
  state.content_commit_lesioned =
      std::getenv("BCC32_LESION_CONTENT_COMMIT") != nullptr;
  // Relational-triple channel: the typed-triple store (subject field claims
  // a slot via CAS; 0xff fill marks every slot empty), counts, retrieval
  // scratch, and the propositional plan. BCC32_LESION_RELATION_TRIPLE
  // skips triple learning AND triple commitment -> exact prior behavior.
  state.relation_triples.allocate(construction::kRelationTripleHashCap);
  state.relation_triple_counts.allocate(construction::kRelationTripleHashCap);
  state.relation_roles.allocate(construction::kRelationRoleHashCap);
  state.relation_role_counts.allocate(construction::kRelationRoleHashCap);
  state.relation_triple_evidence_revision.allocate(
      construction::kRelationTripleHashCap);
  state.witnessed_relation_events.allocate(
      construction::kWitnessedRelationEventCap);
  state.witnessed_relation_event_cursor.allocate(1u);
  state.witnessed_relation_constructions.allocate(
      construction::kWitnessedRelationEventCap);
  state.witnessed_relation_surface_units.allocate(
      construction::kWitnessedRelationEventCap *
      construction::kConstructionMaxTokens);
  state.witnessed_relation_surface_counts.allocate(
      construction::kWitnessedRelationEventCap);
  state.relation_triple_type_total.allocate(state.unit_capacity);
  state.relation_triple_type_mirrored.allocate(state.unit_capacity);
  cuda_require(cudaMemset(state.relation_triple_type_total.get(), 0,
                          state.relation_triple_type_total.bytes()),
               "clear resident relation triple type totals");
  cuda_require(cudaMemset(state.relation_triple_type_mirrored.get(), 0,
                          state.relation_triple_type_mirrored.bytes()),
               "clear resident relation triple mirror totals");
  state.relation_triple_candidates.allocate(
      construction::kRelationTripleCandidateCap);
  state.relation_triple_cursor.allocate(2u);
  state.relation_triple_attempted.allocate(1u);
  state.relation_triple_drops.allocate(1u);
  // Cleared once, at genesis. These accumulate over every ingest so the
  // conservation attempted == sum(counts) + drops holds for the whole store.
  cuda_require(cudaMemset(state.relation_triple_attempted.get(), 0,
                          state.relation_triple_attempted.bytes()),
               "clear resident relation triple attempt total");
  cuda_require(cudaMemset(state.relation_triple_drops.get(), 0,
                          state.relation_triple_drops.bytes()),
               "clear resident relation triple drop total");
  state.relation_probe_support_histogram.allocate(
      construction::kRelationProbeSupportBins);
  state.relation_probe_total.allocate(1u);
  // Cleared once, at genesis: the probe support distribution accumulates
  // over every reply, so one run reports the whole session's probes.
  cuda_require(
      cudaMemset(state.relation_probe_support_histogram.get(), 0,
                 state.relation_probe_support_histogram.bytes()),
      "clear relation probe support histogram");
  cuda_require(cudaMemset(state.relation_probe_total.get(), 0,
                          state.relation_probe_total.bytes()),
               "clear relation probe total");
  state.relation_census_histogram.allocate(
      construction::kRelationProbeSupportBins);
  state.relation_census_total.allocate(1u);
  state.relation_census_source_singletons.allocate(1u);
  // Cleared once, at genesis, on the same terms as the gated probe: the
  // ungated census accumulates over every reply so the two arms cover the
  // same session.
  cuda_require(cudaMemset(state.relation_census_histogram.get(), 0,
                          state.relation_census_histogram.bytes()),
               "clear relation census support histogram");
  cuda_require(cudaMemset(state.relation_census_total.get(), 0,
                          state.relation_census_total.bytes()),
               "clear relation census total");
  cuda_require(cudaMemset(state.relation_census_source_singletons.get(), 0,
                          state.relation_census_source_singletons.bytes()),
               "clear relation census source singleton total");
  // Matched-counterfactual arm, cleared once at genesis on the same terms as
  // the census above so both arms cover the identical session.
  state.relation_counterfactual_histogram.allocate(
      construction::kRelationProbeSupportBins);
  state.relation_counterfactual_total.allocate(1u);
  state.relation_topic_strictly_greater.allocate(1u);
  state.relation_counterfactual_strictly_greater.allocate(1u);
  state.relation_support_ties.allocate(1u);
  cuda_require(cudaMemset(state.relation_counterfactual_histogram.get(), 0,
                          state.relation_counterfactual_histogram.bytes()),
               "clear relation counterfactual histogram");
  cuda_require(cudaMemset(state.relation_counterfactual_total.get(), 0,
                          state.relation_counterfactual_total.bytes()),
               "clear relation counterfactual total");
  cuda_require(cudaMemset(state.relation_topic_strictly_greater.get(), 0,
                          state.relation_topic_strictly_greater.bytes()),
               "clear relation topic strictly greater total");
  cuda_require(
      cudaMemset(state.relation_counterfactual_strictly_greater.get(), 0,
                 state.relation_counterfactual_strictly_greater.bytes()),
      "clear relation counterfactual strictly greater total");
  cuda_require(cudaMemset(state.relation_support_ties.get(), 0,
                          state.relation_support_ties.bytes()),
               "clear relation support tie total");
  state.relation_triple_plan.allocate(construction::kCommitmentCap);
  state.relation_triple_meta.allocate(8u);
  state.relation_cue_scores.allocate(state.unit_capacity);
  state.relation_cue_orders.allocate(state.unit_capacity);
  state.relation_cue_exact.allocate(state.unit_capacity);
  state.proposition_cue_sequence.allocate(kCompositionUnits);
  state.proposition_cue_sequence_count.allocate(1u);
  state.relation_operator_order.allocate(1u);
  state.streaming_cue_bytes.allocate(kStreamingCueCapacity);
  state.streaming_cue_meta.allocate(3u);
  cuda_require(cudaMemset(state.relation_cue_scores.get(), 0,
                          state.relation_cue_scores.bytes()),
               "clear persisted cue identity scores");
  cuda_require(cudaMemset(state.relation_cue_orders.get(), 0xff,
                          state.relation_cue_orders.bytes()),
               "clear persisted cue unit orders");
  cuda_require(cudaMemset(state.relation_cue_exact.get(), 0,
                          state.relation_cue_exact.bytes()),
               "clear persisted exact cue contacts");
  cuda_require(cudaMemset(state.proposition_cue_sequence_count.get(), 0,
                          state.proposition_cue_sequence_count.bytes()),
               "clear ordered proposition cue extent");
  cuda_require(cudaMemset(state.relation_operator_order.get(), 0xff,
                          state.relation_operator_order.bytes()),
               "clear learned cue operator order");
  cuda_require(cudaMemset(state.streaming_cue_bytes.get(), 0,
                          state.streaming_cue_bytes.bytes()),
               "clear streaming cue surface");
  cuda_require(cudaMemset(state.streaming_cue_meta.get(), 0,
                          state.streaming_cue_meta.bytes()),
               "clear streaming cue state");
  cuda_require(cudaMemset(state.relation_triples.get(), 0xff,
                          state.relation_triples.bytes()),
               "clear resident relation triple table");
  cuda_require(cudaMemset(state.relation_triple_counts.get(), 0,
                          state.relation_triple_counts.bytes()),
               "clear resident relation triple counts");
  cuda_require(cudaMemset(state.relation_roles.get(), 0xff,
                          state.relation_roles.bytes()),
               "clear resident relation role cloud");
  cuda_require(cudaMemset(state.relation_role_counts.get(), 0,
                          state.relation_role_counts.bytes()),
               "clear resident relation role cloud counts");
  cuda_require(cudaMemset(state.relation_triple_evidence_revision.get(), 0,
                          state.relation_triple_evidence_revision.bytes()),
               "clear resident relation triple evidence revisions");
  cuda_require(cudaMemset(state.witnessed_relation_events.get(), 0,
                          state.witnessed_relation_events.bytes()),
               "clear witnessed relation events");
  cuda_require(cudaMemset(state.witnessed_relation_event_cursor.get(), 0,
                          state.witnessed_relation_event_cursor.bytes()),
               "clear witnessed relation event cursor");
  cuda_require(cudaMemset(state.witnessed_relation_constructions.get(), 0xff,
                          state.witnessed_relation_constructions.bytes()),
               "clear witnessed relation construction identities");
  cuda_require(cudaMemset(state.witnessed_relation_surface_units.get(), 0xff,
                          state.witnessed_relation_surface_units.bytes()),
               "clear witnessed relation surface units");
  cuda_require(cudaMemset(state.witnessed_relation_surface_counts.get(), 0,
                          state.witnessed_relation_surface_counts.bytes()),
               "clear witnessed relation surface counts");
  cuda_require(cudaMemset(state.relation_triple_meta.get(), 0,
                          state.relation_triple_meta.bytes()),
               "clear resident relation triple plan meta");
  state.relation_triple_lesioned =
      std::getenv("BCC32_LESION_RELATION_TRIPLE") != nullptr;
}

// LESION: emptying the store returns the organism to the legacy generation
// path (the composer only fires when learned constructions exist).
