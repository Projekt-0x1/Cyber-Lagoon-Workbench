inline void learn_resident_constructions(
    AdultState& state, const std::uint32_t* sequence,
    std::uint32_t sequence_count, const std::uint32_t* segment_ids,
    bool efferent_contact = false, std::int32_t efferent_polarity = 0,
    bool outcome_present = true, bool learn_relation_triples = true,
    bool initial_exposure = false,
    const std::uint32_t* contact_learned_request = nullptr) {
  if (state.construction_lesioned || sequence_count == 0u ||
      state.construction_store_count.get() == nullptr || state.unit_count == 0u)
    return;
  const std::uint32_t construction_allocation_capacity =
      initial_exposure ? construction::kInitialConstructionCap
                       : construction::kConstructionCap;
  // LEARN morphological agreement: accumulate suffix-class adjacency counts
  // from the device-resident assimilation sequence. Entirely on-device (the
  // sequence and unit bytes never cross to the host for this signal).
  if (!state.morph_agreement_lesioned && sequence_count > 1u &&
      state.construction_suffix_transitions.get() != nullptr) {
    construction::learn_suffix_transitions_kernel<<<
        (sequence_count + construction::kConstructionBlock - 1u) /
            construction::kConstructionBlock,
        construction::kConstructionBlock>>>(
        sequence, sequence_count, state.unit_lengths.get(),
        state.unit_content.get(), kUnitWords, state.boundary_mask.get(),
        state.construction_suffix_transitions.get());
    cuda_require(cudaGetLastError(), "learn resident suffix-class adjacency");
  }
  cuda_require(cudaMemset(state.construction_role_projection.get(), 0,
                          state.construction_role_projection.bytes()),
               "reset resident construction role projection");
  // CANONICALIZE the role signal: build the unit -> representative map from
  // the device-resident unit bytes so all surface variants of one canonical
  // form pool their context into a single robust role. Entirely on-device
  // (no unit byte crosses to the host for this signal); lesioned -> nullptr
  // map -> bit-exact legacy per-variant roles.
  const std::uint32_t* role_canon = nullptr;
  std::vector<std::uint32_t> host_canon;
  if (!state.role_canon_lesioned &&
      state.construction_role_canon.get() != nullptr) {
    cuda_require(roles::build_role_canonical_map_cuda(
                     state.unit_count, state.unit_lengths.get(),
                     state.unit_content.get(), kUnitWords,
                     state.construction_role_canon_signatures.get(),
                     state.construction_role_canon_keys.get(),
                     state.construction_role_canon_reps.get(),
                     state.construction_role_canon_table_size,
                     state.construction_role_canon.get()),
                 "build resident role canonical map");
    role_canon = state.construction_role_canon.get();
    // The map (unit ids only, no byte content) also feeds the host-published
    // closed-class mask below, exactly like the vitality/role copies the
    // mask publication already makes at HEAD.
    host_canon.resize(state.unit_count);
    cuda_require(cudaMemcpy(host_canon.data(), state.construction_role_canon.get(),
                            host_canon.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read resident role canonical map");
    std::uint32_t representatives = 0u;
    for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
      if (host_canon[unit] == unit) ++representatives;
    }
    std::fprintf(stderr, "role_canon units=%u canonical_forms=%u\n",
                 state.unit_count, representatives);
  }
  cuda_require(roles::derive_structural_roles_cuda(
                   state.unit_count, state.bigrams.get(), state.bigram_counts.get(),
                   state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
                   state.trigram_count, state.online_bigrams.get(),
                   state.online_bigram_counts.get(), state.online_bigram_count,
                   state.online_trigrams.get(), state.online_trigram_counts.get(),
                   state.online_trigram_count, state.construction_role_projection.get(),
                   state.construction_roles.get(), role_canon),
               "derive resident construction roles");
  // Discover the closed-class (function-word) vitality threshold: the
  // frequency of the kConstructionFuncRank-th most vital resident unit.
  std::vector<std::uint32_t> vitality(state.unit_count);
  cuda_require(cudaMemcpy(vitality.data(), state.unit_vitality.get(),
                          vitality.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "read resident vitality for closed-class discovery");
  const std::uint32_t rank = std::min<std::uint32_t>(
      construction::kConstructionFuncRank,
      std::max<std::uint32_t>(1u, state.unit_count / 4u));
  std::nth_element(vitality.begin(), vitality.begin() + (rank - 1u), vitality.end(),
                   [](std::uint32_t a, std::uint32_t b) { return a > b; });
  state.construction_func_threshold = std::max<std::uint32_t>(2u, vitality[rank - 1u]);
  // Publish the closed-class mask -- the ONE authority all construction
  // kernels read for the glue-vs-content decision (vitality rank +
  // role-projection confidence; validated against store yield).
  {
    std::vector<roles::MutableStructuralRole> mask_roles(state.unit_count);
    cuda_require(cudaMemcpy(mask_roles.data(), state.construction_roles.get(),
                            mask_roles.size() * sizeof(roles::MutableStructuralRole),
                            cudaMemcpyDeviceToHost),
                 "read resident roles for closed-class mask");
    std::vector<std::uint32_t> mask_vitality(state.unit_count);
    cuda_require(cudaMemcpy(mask_vitality.data(), state.unit_vitality.get(),
                            mask_vitality.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read resident vitality for closed-class mask");
    std::vector<std::uint32_t> mask(state.unit_count, 0u);
    if (!host_canon.empty()) {
      // CANONICAL-GROUP glue decision: pool vitality across the surface
      // variants of each canonical form so the glue/content partition sees
      // canonical forms, not fragments. Without this, low-vitality
      // punctuation variants of discovered function words ("what?", "The")
      // dodge the glue mask and enter the filler pools as content -- the
      // first role-canon run emitted "the what? of" mid-sentence through
      // exactly that hole. Same discovered rank statistic, applied to the
      // pooled counts; no authored word list.
      std::vector<unsigned long long> group_vitality(state.unit_count, 0ull);
      for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit)
        group_vitality[host_canon[unit]] += mask_vitality[unit];
      std::vector<unsigned long long> group_values;
      for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
        if (host_canon[unit] == unit && group_vitality[unit] != 0ull)
          group_values.push_back(group_vitality[unit]);
      }
      if (!group_values.empty()) {
        const std::uint32_t group_rank = std::min<std::uint32_t>(
            construction::kConstructionFuncRank,
            std::max<std::uint32_t>(
                1u, static_cast<std::uint32_t>(group_values.size() / 4u)));
        std::nth_element(group_values.begin(), group_values.begin() + (group_rank - 1u),
                         group_values.end(),
                         [](unsigned long long a, unsigned long long b) { return a > b; });
        const unsigned long long group_threshold =
            std::max<unsigned long long>(2ull, group_values[group_rank - 1u]);
        state.construction_func_threshold = static_cast<std::uint32_t>(
            std::min<unsigned long long>(group_threshold, 0xffffffffull));
        for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
          if (group_vitality[host_canon[unit]] >= group_threshold)
            mask[unit] = 1u;
        }
      }
    } else {
      for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
        // Frequency rank alone: glue is stored LITERALLY, so its role
        // confidence is irrelevant (only slot positions need roles). The
        // confidence requirement was tried and collapsed the glue vocabulary
        // to 14 units (top-frequency units often have cancelling projections).
        if (mask_vitality[unit] >= state.construction_func_threshold)
          mask[unit] = 1u;
      }
    }
    cuda_require(cudaMemcpy(state.construction_closed_class_mask.get(), mask.data(),
                            mask.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice),
                 "publish resident closed-class mask");
  }
  // NEIGHBOR-ENTROPY glue partition: overwrite the frequency-rank mask with
  // the context-dispersion decision. Frequency rank absorbs high-frequency
  // CONTENT words ("work", "economic") into the glue set, so they can never
  // fill content slots and the composer reaches for the wrong word ("human
  // worm" for "human work"). What actually distinguishes glue is FREE
  // COMBINATION: high mean left/right neighbor-entropy of the canonical
  // group's resident bigram distribution. Computed and published fully
  // ON-DEVICE from the resident neighbor counts (no count or byte crosses to
  // the host for this signal; the host copies below are stderr diagnostics
  // only). Lesion BCC32_LESION_ENTROPY_GLUE -> the frequency mask published
  // above stands untouched.
  if (!state.entropy_glue_lesioned) {
    const std::uint32_t entropy_table_size = roles::entropy_glue_table_size(
        state.bigram_count, state.online_bigram_count);
    DeviceArray<unsigned long long> pair_keys(entropy_table_size);
    DeviceArray<unsigned long long> pair_counts(entropy_table_size);
    DeviceArray<unsigned long long> right_totals(state.unit_count);
    DeviceArray<unsigned long long> left_totals(state.unit_count);
    DeviceArray<double> right_plogp(state.unit_count);
    DeviceArray<double> left_plogp(state.unit_count);
    DeviceArray<std::uint32_t> entropy_histogram(roles::kEntropyGlueHistogramBins);
    DeviceArray<int> max_entropy_bits(1u);
    DeviceArray<float> entropy_mean(state.unit_count);
    DeviceArray<float> entropy_min(state.unit_count);
    DeviceArray<float> entropy_cutoff(1u);
    cuda_require(
        roles::discover_entropy_glue_cuda<BigramKey>(
            state.unit_count, state.bigrams.get(), state.bigram_counts.get(),
            state.bigram_count, state.online_bigrams.get(),
            state.online_bigram_counts.get(), state.online_bigram_count,
            role_canon, pair_keys.get(), pair_counts.get(), entropy_table_size,
            right_totals.get(), left_totals.get(), right_plogp.get(),
            left_plogp.get(), entropy_histogram.get(), max_entropy_bits.get(),
            entropy_mean.get(), entropy_min.get(), entropy_cutoff.get(),
            state.construction_closed_class_mask.get()),
        "discover resident neighbor-entropy glue mask");
    // Diagnostics ONLY from here on -- the mask authority was written by the
    // device kernel above and is never re-published from these copies. The
    // synchronous cudaMemcpy immediately below already blocks on the legacy
    // default stream, making an explicit cudaDeviceSynchronize() here
    // redundant (0X1-284).
    float host_cutoff = 0.0f;
    cuda_require(cudaMemcpy(&host_cutoff, entropy_cutoff.get(), sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "read entropy glue cutoff for diagnostics");
    std::vector<std::uint32_t> host_mask(state.unit_count);
    cuda_require(cudaMemcpy(host_mask.data(),
                            state.construction_closed_class_mask.get(),
                            host_mask.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read entropy glue mask for diagnostics");
    std::uint32_t glue_units = 0u;
    std::uint32_t glue_groups = 0u;
    for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
      if (host_mask[unit] == 0u) continue;
      ++glue_units;
      if (host_canon.empty() || host_canon[unit] == unit) ++glue_groups;
    }
    std::fprintf(stderr,
                 "entropy_glue cutoff=%.3f glue_units=%u glue_groups=%u\n",
                 host_cutoff, glue_units, glue_groups);
    if (std::getenv("BCC32_TRACE_ENTROPY_GLUE") != nullptr) {
      std::vector<float> host_mean(state.unit_count);
      std::vector<float> host_min(state.unit_count);
      cuda_require(cudaMemcpy(host_mean.data(), entropy_mean.get(),
                              host_mean.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "read entropy means for trace");
      cuda_require(cudaMemcpy(host_min.data(), entropy_min.get(),
                              host_min.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "read entropy minima for trace");
      std::vector<std::uint32_t> trace_lengths(state.unit_count);
      cuda_require(cudaMemcpy(trace_lengths.data(), state.unit_lengths.get(),
                              trace_lengths.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read unit lengths for trace");
      std::vector<std::uint32_t> trace_content(
          static_cast<std::size_t>(state.unit_count) * kUnitWords);
      cuda_require(cudaMemcpy(trace_content.data(), state.unit_content.get(),
                              trace_content.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read unit content for trace");
      std::vector<std::uint32_t> ordered;
      for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
        if (host_mean[unit] < 0.0f) continue;
        if (!host_canon.empty() && host_canon[unit] != unit) continue;
        ordered.push_back(unit);
      }
      std::sort(ordered.begin(), ordered.end(),
                [&](std::uint32_t a, std::uint32_t b) {
                  return host_mean[a] > host_mean[b];
                });
      const std::size_t trace_limit = std::min<std::size_t>(ordered.size(), 96u);
      for (std::size_t rank = 0u; rank < trace_limit; ++rank) {
        const std::uint32_t unit = ordered[rank];
        char word[48u] = {};
        const std::uint32_t length =
            std::min<std::uint32_t>(trace_lengths[unit], 47u);
        for (std::uint32_t offset = 0u; offset < length; ++offset) {
          const std::uint32_t value =
              (trace_content[static_cast<std::size_t>(unit) * kUnitWords +
                             offset / 4u] >>
               ((offset % 4u) * 8u)) &
              0xffu;
          word[offset] = static_cast<char>(
              value >= 0x20u && value < 0x7fu ? value : '?');
        }
        std::fprintf(stderr,
                     "entropy_glue_trace rank=%zu unit=%u word='%s'"
                     " mean=%.3f min=%.3f glue=%u\n",
                     rank, unit, word, host_mean[unit], host_min[unit],
                     host_mask[unit]);
      }
    }
  }
  // RELATIONAL-TRIPLE LEARN: scan the device-resident assimilation sequence
  // for [content A][glue run K][content B] bridges under the JUST-PUBLISHED
  // glue partition and accumulate typed triples (A, K, B) into the resident
  // table. Entirely on-device: the sequence, the mask, and the table never
  // cross to the host for this signal (the host reads back two occupancy
  // integers for stderr only). The connective inventory is the discovered
  // entropy-glue set -- no authored connective, no parser.
  if (learn_relation_triples && !state.relation_triple_lesioned &&
      sequence_count > 2u &&
      state.relation_triples.get() != nullptr) {
    DeviceArray<std::uint32_t> contact_relation_event_count(1u);
    cuda_require(cudaMemset(contact_relation_event_count.get(), 0,
                            contact_relation_event_count.bytes()),
                 "reset per-contact relation event count");
    // Advance once before every event consumer. The retained triple revision
    // and the construction witness must name the same physical contact.
    if (state.surface_unit_population.get() != nullptr &&
        state.proposition_ordered_evidence_revision.get() != nullptr) {
      ordered_relation::advance_contact_revision_kernel<<<1u, 1u>>>(
          state.proposition_ordered_evidence_revision.get());
    }
    construction::learn_relation_triples_kernel<<<
        (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
        sequence, sequence_count, segment_ids,
        state.construction_closed_class_mask.get(),
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.qonset_evidence_revision.get(),
        state.relation_triples.get(), state.relation_triple_counts.get(),
        state.relation_triple_evidence_revision.get(),
        state.relation_roles.get(), state.relation_role_counts.get(),
        state.proposition_ordered_evidence_revision.get(),
        state.witnessed_relation_events.get(),
        state.witnessed_relation_event_cursor.get(),
        state.witnessed_relation_constructions.get(),
        state.witnessed_relation_surface_units.get(),
        state.witnessed_relation_surface_counts.get(),
        contact_relation_event_count.get(), contact_learned_request,
        state.qterm_count.get(), state.unit_lengths.get(),
        state.unit_content.get(), kUnitWords,
        state.construction_closure_bytes.get(),
        state.construction_closure_count, state.boundary_mask.get(),
        state.unit_count, state.relation_triple_attempted.get(),
        state.relation_triple_drops.get());
    cuda_require(cudaGetLastError(), "learn resident relation triples");
    // learn_question_gap_fields_kernel used to launch HERE; do not re-add it.
    // Its closure gates are only written later in this same function.
    if (state.surface_unit_population.get() != nullptr &&
        state.proposition_ordered_evidence_revision.get() != nullptr) {
      const auto resident_tissue = resident_proposition_tissue_view(state);
      DeviceArray<proposition_tissue::OrderedRoleBindingEvidence> shadow_bindings(
          state.proposition_ordered_bindings.size());
      DeviceArray<proposition_tissue::TissueScalars> shadow_scalars(1u);
      DeviceArray<proposition_tissue::OrderedBindingAdmissionState>
          shadow_admission(1u);
      DeviceArray<ordered_relation::AssimilationReceipt> preflight_receipt(1u);
      DeviceArray<ordered_relation::AssimilationReceipt> committed_receipt(1u);
      cuda_require(cudaMemcpy(shadow_bindings.get(),
                              state.proposition_ordered_bindings.get(),
                              state.proposition_ordered_bindings.bytes(),
                              cudaMemcpyDeviceToDevice),
                   "shadow ordered relation bindings");
      cuda_require(cudaMemcpy(shadow_scalars.get(), state.proposition_scalars.get(),
                              state.proposition_scalars.bytes(),
                              cudaMemcpyDeviceToDevice),
                   "shadow ordered relation scalars");
      cuda_require(cudaMemcpy(
                       shadow_admission.get(),
                       state.proposition_ordered_binding_admission.get(),
                       state.proposition_ordered_binding_admission.bytes(),
                       cudaMemcpyDeviceToDevice),
                   "shadow ordered relation admission");
      auto shadow_tissue = resident_tissue;
      shadow_tissue.ordered_bindings = shadow_bindings.get();
      shadow_tissue.scalars = shadow_scalars.get();
      shadow_tissue.ordered_binding_admission = shadow_admission.get();
      // The relation-triple store is an aggregate recurrence cache. It has
      // already counted this complete contact on device, but it is not an
      // episodic store. The ordered tissue below retains every exact context;
      // recurrence and independent contexts separately gate generalized
      // discourse/causal use. Shadow authorization makes all admitted
      // mutations capacity/mass atomic without a device-to-host decision.
      const ordered_relation::RelationObservationView relation_observations{
          state.relation_triples.get(), state.relation_triple_counts.get(),
          ordered_relation::kMinimumRecurrentObservations};
      ordered_relation::assimilate_relation_events_kernel<<<1u, 32u>>>(
          shadow_tissue, sequence, sequence_count, segment_ids,
          state.construction_closed_class_mask.get(),
          resident_surface_population_view(state),
          state.proposition_ordered_evidence_revision.get(),
          efferent_contact ? 1u : 0u, efferent_polarity,
          outcome_present ? 1u : 0u, nullptr, preflight_receipt.get(),
          relation_observations, contact_relation_event_count.get());
      ordered_relation::commit_assimilation_shadow_kernel<<<
          (state.proposition_ordered_bindings.size() + kBlock - 1u) / kBlock,
          kBlock>>>(
          resident_tissue, shadow_tissue, preflight_receipt.get(),
          committed_receipt.get());
      accumulate_ordered_assimilation_receipt_kernel<<<1u, 1u>>>(
          committed_receipt.get(),
          state.proposition_construction_association.get());
      cuda_require(cudaGetLastError(),
                   "assimilate exact ordered relation events");
      // This receipt is observer-only.  Ordinary learning must remain entirely
      // device resident: no per-contact device-to-host synchronization and no
      // unsolicited stderr traffic on the production path.
      const char* ordered_relation_diag =
          std::getenv("BCC32_ORDERED_RELATION_DIAG");
      if (ordered_relation_diag != nullptr && ordered_relation_diag[0] == '1') {
        ordered_relation::AssimilationReceipt host_receipt{};
        cuda_require(cudaMemcpy(&host_receipt, committed_receipt.get(),
                                sizeof(host_receipt), cudaMemcpyDeviceToHost),
                     "read ordered relation assimilation receipt");
        std::fprintf(
            stderr,
            "ordered_relation_assimilation events=%llu accepted=%llu replay=%llu"
            " invalid=%llu incomplete=%llu capacity=%llu mass=%llu"
            " event_overflow=%llu whole_abstain=%llu below_recurrence=%llu"
            " recurrent=%llu context_replay=%llu generalized=%llu"
            // The four-way split of the incomplete_population route. The
            // aggregate pair cannot name which route fired, because invalid
            // and incomplete_population are incremented together in one
            // branch, so a whole contact can abstain without saying why.
            // These four partition incomplete_population exactly while the
            // default-invalid route stays silent.
            " pop_missing=%llu pop_primary=%llu pop_context=%llu"
            " pop_distinct=%llu\n",
            static_cast<unsigned long long>(host_receipt.events),
            static_cast<unsigned long long>(host_receipt.accepted),
            static_cast<unsigned long long>(host_receipt.replays),
            static_cast<unsigned long long>(host_receipt.invalid),
            static_cast<unsigned long long>(host_receipt.incomplete_population),
            static_cast<unsigned long long>(host_receipt.capacity_overflow),
            static_cast<unsigned long long>(host_receipt.insufficient_mass),
            static_cast<unsigned long long>(host_receipt.event_overflow),
            static_cast<unsigned long long>(host_receipt.whole_contact_abstained),
            static_cast<unsigned long long>(host_receipt.below_recurrence),
            static_cast<unsigned long long>(host_receipt.recurrent),
            static_cast<unsigned long long>(host_receipt.context_replays),
            static_cast<unsigned long long>(host_receipt.generalized),
            static_cast<unsigned long long>(host_receipt.population_missing_view),
            static_cast<unsigned long long>(host_receipt.population_primary_unit),
            static_cast<unsigned long long>(host_receipt.population_context_unit),
            static_cast<unsigned long long>(
                host_receipt.population_context_distinct));
      }
    }
    construction::learn_qonset_kernel<<<
        (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
        sequence, sequence_count, segment_ids, state.unit_lengths.get(),
        state.unit_content.get(), kUnitWords,
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.qonset_count.get(),
        state.proposition_ordered_evidence_revision.get(),
        state.qonset_evidence_revision.get(), state.unit_count);
    cuda_require(cudaGetLastError(), "learn resident question openers");
    // QONSET OCCUPANCY. The question-gap learner skips every position whose
    // opener has qonset_evidence_revision == 0, and NOTHING in the tree reports
    // whether that table is populated -- learn_relation_triples_kernel is even
    // handed it and (void)s it. Without this count, "the gap table is empty"
    // cannot be separated from "no opener was ever recognised": both present
    // identically as rows=0. The stored value is a contact revision, not a
    // flag, so it is also pinnable at 0 if advance_contact_revision_kernel
    // never ran -- counting nonzero entries distinguishes that too.
    if (std::getenv("BCC32_CLOSURE_DIAG") != nullptr &&
        state.qonset_evidence_revision.get() != nullptr && state.unit_count != 0u) {
      std::vector<std::uint64_t> host_qonset(state.unit_count, 0u);
      cuda_require(cudaMemcpy(host_qonset.data(),
                              state.qonset_evidence_revision.get(),
                              host_qonset.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost),
                   "read qonset occupancy");
      std::uint32_t occupied = 0u;
      std::uint64_t peak = 0u;
      for (std::uint64_t value : host_qonset) {
        occupied += value != 0u ? 1u : 0u;
        peak = value > peak ? value : peak;
      }
      std::fprintf(stderr,
                   "qonset_occupancy units=%u nonzero=%u peak_revision=%llu\n",
                   state.unit_count, occupied,
                   static_cast<unsigned long long>(peak));
    }
    // Recompute the learned per-type directionality statistic from the
    // whole store (mirror-attested mass per connective).
    cuda_require(cudaMemset(state.relation_triple_type_total.get(), 0,
                            state.relation_triple_type_total.bytes()),
                 "reset relation triple type totals");
    cuda_require(cudaMemset(state.relation_triple_type_mirrored.get(), 0,
                            state.relation_triple_type_mirrored.bytes()),
                 "reset relation triple mirror totals");
    construction::accumulate_triple_mirror_stats_kernel<<<
        (construction::kRelationTripleHashCap + kBlock - 1u) / kBlock,
        kBlock>>>(state.relation_triples.get(),
                  state.relation_triple_counts.get(),
                  state.relation_triple_type_total.get(),
                  state.relation_triple_type_mirrored.get());
    cuda_require(cudaGetLastError(), "accumulate relation triple mirror stats");
    cuda_require(cudaMemset(state.relation_triple_cursor.get(), 0,
                            state.relation_triple_cursor.bytes()),
                 "clear relation triple occupancy stats");
    construction::count_relation_triples_kernel<<<
        (construction::kRelationTripleHashCap + kBlock - 1u) / kBlock,
        kBlock>>>(state.relation_triple_counts.get(),
                  construction::kRelationTripleHashCap,
                  state.relation_triple_cursor.get());
    cuda_require(cudaGetLastError(), "count resident relation triples");
    std::uint32_t triple_stats[2] = {};
    cuda_require(cudaMemcpy(triple_stats, state.relation_triple_cursor.get(),
                            sizeof(triple_stats), cudaMemcpyDeviceToHost),
                 "read resident relation triple occupancy");
    // Summed counts are the mass that landed; drops are the mass the probe
    // window refused. Their sum must equal the attempts, or the store has
    // lost triples through a path nothing counts.
    std::uint32_t triple_attempted = 0u;
    std::uint32_t triple_drops = 0u;
    cuda_require(cudaMemcpy(&triple_attempted,
                            state.relation_triple_attempted.get(),
                            sizeof(triple_attempted), cudaMemcpyDeviceToHost),
                 "read resident relation triple attempt total");
    cuda_require(cudaMemcpy(&triple_drops, state.relation_triple_drops.get(),
                            sizeof(triple_drops), cudaMemcpyDeviceToHost),
                 "read resident relation triple drop total");
    std::fprintf(stderr,
                 "relation_triple_learn distinct=%u events=%u attempted=%u "
                 "drops=%u conserved=%u\n",
                 triple_stats[0], triple_stats[1], triple_attempted,
                 triple_drops,
                 triple_attempted == triple_stats[1] + triple_drops ? 1u : 0u);
    // Opt-in discovery diagnostic (BCC32_TRIPLE_DIAG): which connective
    // TYPES did the stream teach? Host reads back ids/counts and unit bytes
    // for stderr display only -- no content decision is made here.
    if (std::getenv("BCC32_TRIPLE_DIAG") != nullptr && triple_stats[0] != 0u) {
      std::vector<construction::RelationTriple> host_table(
          construction::kRelationTripleHashCap);
      std::vector<std::uint32_t> host_counts(
          construction::kRelationTripleHashCap);
      cuda_require(cudaMemcpy(host_table.data(), state.relation_triples.get(),
                              host_table.size() *
                                  sizeof(construction::RelationTriple),
                              cudaMemcpyDeviceToHost),
                   "read relation triple table for diagnostics");
      cuda_require(cudaMemcpy(host_counts.data(),
                              state.relation_triple_counts.get(),
                              host_counts.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read relation triple counts for diagnostics");
      std::unordered_map<std::uint32_t, std::uint64_t> connective_mass;
      std::unordered_map<std::uint32_t, std::uint32_t> connective_kinds;
      for (std::size_t i = 0u; i < host_table.size(); ++i) {
        if (host_counts[i] == 0u) continue;
        connective_mass[host_table[i].connective] += host_counts[i];
        connective_kinds[host_table[i].connective] += 1u;
      }
      std::vector<std::pair<std::uint32_t, std::uint64_t>> ranked(
          connective_mass.begin(), connective_mass.end());
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
      const std::size_t shown = std::min<std::size_t>(ranked.size(), 12u);
      for (std::size_t r = 0u; r < shown; ++r) {
        const std::uint32_t unit = ranked[r].first;
        std::uint32_t length = 0u;
        cuda_require(cudaMemcpy(&length, state.unit_lengths.get() + unit,
                                sizeof(length), cudaMemcpyDeviceToHost),
                     "read connective unit length");
        std::uint32_t words[kUnitWords] = {};
        cuda_require(cudaMemcpy(words,
                                state.unit_content.get() +
                                    static_cast<std::size_t>(unit) * kUnitWords,
                                sizeof(words), cudaMemcpyDeviceToHost),
                     "read connective unit bytes");
        char text[25] = {};
        const std::uint32_t bounded = std::min<std::uint32_t>(length, 24u);
        for (std::uint32_t offset = 0u; offset < bounded; ++offset) {
          const std::uint8_t byte = static_cast<std::uint8_t>(
              words[offset / 4u] >> ((offset % 4u) * 8u));
          text[offset] =
              byte >= 32u && byte < 127u ? static_cast<char>(byte) : '.';
        }
        std::uint32_t type_total = 0u;
        std::uint32_t type_mirrored = 0u;
        cuda_require(cudaMemcpy(&type_total,
                                state.relation_triple_type_total.get() + unit,
                                sizeof(type_total), cudaMemcpyDeviceToHost),
                     "read connective type total");
        cuda_require(
            cudaMemcpy(&type_mirrored,
                       state.relation_triple_type_mirrored.get() + unit,
                       sizeof(type_mirrored), cudaMemcpyDeviceToHost),
            "read connective mirror total");
        std::fprintf(stderr,
                     "relation_triple_type rank=%zu unit=%u word='%s'"
                     " events=%llu distinct=%u mirror=%u/%u%s\n",
                     r, unit, text,
                     static_cast<unsigned long long>(ranked[r].second),
                     connective_kinds[unit], type_mirrored, type_total,
                     (static_cast<unsigned long long>(type_mirrored)
                      << construction::kRelationTripleMirrorShift) >=
                             static_cast<unsigned long long>(type_total)
                         ? " SYMMETRIC"
                         : " directional");
      }
    }
  }
  cuda_require(cudaMemset(state.construction_role_population.get(), 0,
                          state.construction_role_population.bytes()),
               "reset resident construction role population");
  construction::count_role_population_kernel<<<
      (state.unit_count + construction::kConstructionBlock - 1u) /
          construction::kConstructionBlock,
      construction::kConstructionBlock>>>(
      state.construction_roles.get(), state.unit_count,
      state.construction_role_population.get());
  cuda_require(cudaGetLastError(), "count resident construction role population");
  std::vector<std::uint32_t> role_population(roles::kStructuralRoleCount);
  cuda_require(cudaMemcpy(role_population.data(),
                          state.construction_role_population.get(),
                          role_population.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "read resident construction role population");
  std::vector<std::uint32_t> occupied_role_population;
  for (const std::uint32_t population : role_population) {
    if (population != 0u) occupied_role_population.push_back(population);
  }
  if (occupied_role_population.empty()) return;
  const std::size_t compact_role_rank = occupied_role_population.size() / 4u;
  std::nth_element(occupied_role_population.begin(),
                   occupied_role_population.begin() + compact_role_rank,
                   occupied_role_population.end());
  state.construction_role_population_cutoff =
      occupied_role_population[compact_role_rank];

  std::uint32_t last_segment = 0u;
  cuda_require(cudaMemcpy(&last_segment, segment_ids + sequence_count - 1u,
                          sizeof(last_segment), cudaMemcpyDeviceToHost),
               "read construction episode extent");
  const std::uint32_t episode_count = last_segment + 1u;
  DeviceArray<std::uint32_t> episode_offsets(episode_count + 1u);
  scatter_surface_episode_offsets_kernel<<<
      (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
      segment_ids, sequence_count, episode_offsets.get());
  // The synchronous cudaMemcpy immediately below already blocks on the
  // legacy default stream, making an explicit cudaDeviceSynchronize() here
  // redundant (0X1-284).

  // DISCOVER sentence-closure bytes from the resident stream itself: a
  // terminator is a unit-terminal byte whose FOLLOWING units start with
  // globally-unusual first bytes (sentence starts are drawn from a rare
  // first-byte population). Purely statistical -- no authored byte values,
  // and independent of the episode machinery (whose own closure guess is a
  // clause-level ':'/';' on this corpus).
  {
    std::vector<std::uint32_t> host_sequence(sequence_count);
    cuda_require(cudaMemcpy(host_sequence.data(), sequence,
                            host_sequence.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read assimilation sequence for closure discovery");
    std::vector<std::uint32_t> host_lengths(state.unit_count);
    cuda_require(cudaMemcpy(host_lengths.data(), state.unit_lengths.get(),
                            host_lengths.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read unit lengths for closure discovery");
    std::vector<std::uint32_t> host_content(
        static_cast<std::size_t>(state.unit_count) * kUnitWords);
    cuda_require(cudaMemcpy(host_content.data(), state.unit_content.get(),
                            host_content.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read unit content for closure discovery");
    std::vector<std::uint32_t> host_boundary_mask(256u);
    cuda_require(cudaMemcpy(host_boundary_mask.data(), state.boundary_mask.get(),
                            host_boundary_mask.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read boundary mask for closure discovery");
    auto unit_byte = [&](std::uint32_t unit, std::uint32_t offset) -> std::uint32_t {
      const std::uint32_t word =
          host_content[static_cast<std::size_t>(unit) * kUnitWords + offset / 4u];
      return (word >> ((offset % 4u) * 8u)) & 0xffu;
    };
    auto terminal_byte = [&](std::uint32_t unit) -> int {
      const std::uint32_t length = host_lengths[unit];
      if (length == 0u) return -1;
      const std::uint32_t trailing = unit_byte(unit, length - 1u);
      for (std::uint32_t back = 0u; back < length; ++back) {
        const std::uint32_t value = unit_byte(unit, length - 1u - back);
        if (value != trailing) return static_cast<int>(value);
      }
      return static_cast<int>(trailing);
    };
    std::vector<std::uint64_t> first_histogram(256u, 0u);
    std::vector<std::uint64_t> byte_total(256u, 0u);
    for (std::uint32_t position = 0u; position < sequence_count; ++position) {
      const std::uint32_t unit = host_sequence[position];
      const std::uint32_t length = host_lengths[unit];
      if (length != 0u) ++first_histogram[unit_byte(unit, 0u)];
      for (std::uint32_t offset = 0u; offset < length; ++offset)
        ++byte_total[unit_byte(unit, offset)];
    }
    std::vector<std::uint64_t> terminal_total(256u, 0u);
    std::vector<std::uint64_t> terminal_rare_next(256u, 0u);
    const double rare_share = 0.02;
    for (std::uint32_t position = 0u; position + 1u < sequence_count; ++position) {
      const std::uint32_t unit = host_sequence[position];
      const std::uint32_t next = host_sequence[position + 1u];
      const int terminal = terminal_byte(unit);
      if (terminal < 0 || host_lengths[next] == 0u) continue;
      const std::uint32_t next_first = unit_byte(next, 0u);
      ++terminal_total[static_cast<std::uint32_t>(terminal)];
      const double next_share =
          static_cast<double>(first_histogram[next_first]) / sequence_count;
      if (next_share < rare_share)
        ++terminal_rare_next[static_cast<std::uint32_t>(terminal)];
    }
    const std::uint64_t count_floor =
        std::max<std::uint64_t>(8u, sequence_count / 1000u);
    // Closure is an evidence tournament, not an absolute fraction gate. Small
    // resident streams often make every recurrent terminal look maximally rare;
    // multiplying their median then creates an impossible threshold above one.
    // Rank by the observed rare-next ratio and use recurrence mass as the tie
    // break. No byte identity enters the choice.
    std::uint32_t discovered[4] = {};
    std::uint32_t discovered_count = 0u;
    for (std::uint32_t pick = 0u; pick < 4u; ++pick) {
      std::uint32_t best = 256u;
      std::uint64_t best_rare = 0u;
      std::uint64_t best_total = 1u;
      for (std::uint32_t byte = 0u; byte < 256u; ++byte) {
        if (terminal_total[byte] < count_floor) continue;
        if (terminal_rare_next[byte] == 0u) continue;
        // A closure is a boundary phenomenon, not merely a byte that happens
        // to end frequent lexical units.  Keep only candidates whose observed
        // terminal mass outweighs their own interior mass.  This comparison is
        // learned from the same contact and names no literal delimiter.
        const std::uint64_t interior_mass =
            byte_total[byte] > terminal_total[byte]
                ? byte_total[byte] - terminal_total[byte]
                : 0u;
        if (terminal_total[byte] <= interior_mass) continue;
        bool taken = false;
        for (std::uint32_t k = 0u; k < discovered_count; ++k)
          taken |= discovered[k] == byte;
        if (taken) continue;
        const unsigned __int128 lhs =
            static_cast<unsigned __int128>(terminal_rare_next[byte]) * best_total;
        const unsigned __int128 rhs =
            static_cast<unsigned __int128>(best_rare) * terminal_total[byte];
        if (best == 256u || lhs > rhs ||
            (lhs == rhs && terminal_total[byte] > best_total)) {
          best_rare = terminal_rare_next[byte];
          best_total = terminal_total[byte];
          best = byte;
        }
      }
      if (best == 256u) break;
      discovered[discovered_count++] = best;
    }
    // Absence of closure evidence in one small online contact is not evidence
    // against the closure field acquired from the resident corpus. Preserve
    // that long-term matter until a contact contains positive replacement
    // evidence; otherwise the first short sentence erases stream completion.
    // WHICH GATE STILL BLOCKS THE QUESTION-GAP LEARNER. Moving that kernel
    // below this assignment did NOT populate its table -- measured
    // `resident_plan_question_gap rows=0 cue_rows=0` after the move -- so the
    // launch ordering was necessary but not sufficient. The learner is gated on
    // BOTH `closure_count != 0` and a non-empty `qonset_evidence_revision`, and
    // this assignment is itself conditional: a contact that discovers no
    // closure bytes leaves the count at its previous value, which on a first
    // call is the literal 0u from its declaration. Host-side fprintf to stderr
    // is safe here; only DEVICE printf collides with the duplex frame channel.
    if (std::getenv("BCC32_CLOSURE_DIAG") != nullptr)
      std::fprintf(stderr,
                   "closure_discovery discovered_count=%u prior_count=%u\n",
                   discovered_count, state.construction_closure_count);
    if (discovered_count != 0u) {
      state.construction_closure_count = discovered_count;
      cuda_require(cudaMemcpy(state.construction_closure_bytes.get(), discovered,
                              sizeof(discovered), cudaMemcpyHostToDevice),
                   "publish resident construction closure bytes");
      // The construction field is the stricter learned boundary authority: it
      // distinguishes terminal use from lexical interior use.  Publish its
      // opaque result back to the stream segmentation field so later contacts
      // cannot keep using a stale, weaker boundary hypothesis.
      cuda_require(cudaMemset(state.closure_bytes.get(), 0,
                              state.closure_bytes.bytes()),
                   "clear superseded stream closure bytes");
      cuda_require(cudaMemcpy(state.closure_bytes.get(),
                              state.construction_closure_bytes.get(),
                              state.closure_bytes.bytes(),
                              cudaMemcpyDeviceToDevice),
                   "publish learned stream closure bytes");
    }

    // Filler-terminal mask: a byte qualifies as a filler ending only when it
    // is word-interior-dominant in the resident stream (letters), excluding
    // clause punctuation like ','/';'/':'/'.' purely by statistics.
    if (discovered_count != 0u) {
      std::vector<std::uint64_t> byte_terminal(256u, 0u);
      for (std::uint32_t position = 0u; position < sequence_count; ++position) {
        const std::uint32_t unit = host_sequence[position];
        const std::uint32_t length = host_lengths[unit];
        const int terminal = terminal_byte(unit);
        if (terminal >= 0) ++byte_terminal[static_cast<std::uint32_t>(terminal)];
      }
      std::vector<std::uint32_t> terminal_mask(256u, 0u);
      for (std::uint32_t byte = 0u; byte < 256u; ++byte) {
        if (byte_total[byte] == 0u) continue;
        const double interior_share =
            static_cast<double>(byte_total[byte] - byte_terminal[byte]) /
            static_cast<double>(byte_total[byte]);
        if (interior_share >= 0.15) terminal_mask[byte] = 1u;
      }
      cuda_require(cudaMemcpy(state.construction_filler_terminal_mask.get(),
                              terminal_mask.data(),
                              terminal_mask.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice),
                   "publish resident filler terminal mask");
    }

    // Initial-form mask: units that occur predominantly right after a
    // discovered closure are sentence-initial operator forms ('What', 'Why',
    // 'But'); binding bars them from non-initial slots.
    if (discovered_count != 0u) {
      std::vector<std::uint32_t> unit_total(state.unit_count, 0u);
      std::vector<std::uint32_t> unit_initial(state.unit_count, 0u);
      bool previous_closed = true;
      for (std::uint32_t position = 0u; position < sequence_count; ++position) {
        const std::uint32_t unit = host_sequence[position];
        if (unit < state.unit_count) {
          ++unit_total[unit];
          if (previous_closed) ++unit_initial[unit];
        }
        const int terminal = terminal_byte(unit);
        previous_closed = false;
        if (terminal >= 0) {
          for (std::uint32_t k = 0u; k < discovered_count; ++k)
            previous_closed |= discovered[k] == static_cast<std::uint32_t>(terminal);
        }
      }
      std::vector<std::uint32_t> initial_mask(state.unit_count, 0u);
      for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
        if (unit_total[unit] >= 4u &&
            5u * unit_initial[unit] >= 4u * unit_total[unit])
          initial_mask[unit] = 1u;
      }
      cuda_require(cudaMemcpy(state.construction_initial_form_mask.get(),
                              initial_mask.data(),
                              initial_mask.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice),
                   "publish resident initial-form mask");
    }
  }

  construction::learn_constructions_kernel<<<
      (episode_count + construction::kConstructionBlock - 1u) /
          construction::kConstructionBlock,
      construction::kConstructionBlock>>>(
      sequence, episode_offsets.get(), episode_count, state.construction_roles.get(),
      state.construction_closed_class_mask.get(),
      state.unit_lengths.get(), state.unit_content.get(), kUnitWords,
      state.boundary_mask.get(),
      state.construction_closure_bytes.get(), state.construction_closure_count,
      state.construction_tokens.get(), state.construction_lengths.get(),
      state.construction_slot_counts.get(), state.construction_supports.get(),
      state.construction_slot_units.get(), state.construction_slot_masses.get(),
      state.construction_slot_totals.get(), state.construction_slot_overflow.get(),
      state.construction_hash_slots.get(), state.construction_store_count.get(),
      state.proposition_ordered_evidence_revision.get(),
      state.construction_evidence_revision.get(),
      state.construction_origin_revision.get(),
      construction_allocation_capacity);
  cuda_require(cudaGetLastError(), "learn resident constructions");
  // learn_question_gap_fields_kernel used to launch HERE; do not re-add it.
  // RETIRED FROM PRODUCTION (2026-08-14, 0X1-156): its differences==1u
  // acquisition rule was measured to fire ZERO times over 12,000+ real
  // question/answer pairs across four corpora (including Socratic dialogue,
  // chosen to maximize question/answer parallelism), on top of the
  // dead-by-construction closure-ordering issue noted historically below --
  // mass sits entirely at 3-4 differing coordinates, never exactly one, so
  // even a correctly-ordered launch would not have helped. requested_field is
  // now derived live, per query, from the resident relation store by the
  // causal-compatibility vote wired into form_witnessed_relation_plan_kernel
  // (bcc32_resident_relation_answer_side_causal_compatibility.cuh). The
  // kernel definition and its own synthetic-fixture contract
  // (bcc32_cuda_learned_question_gap_contract.cu) are intentionally
  // untouched -- only this production call site is bypassed.
  //
  // Historical note on the closure-ordering issue this call site was moved
  // to solve, kept for context: the kernel is hard-gated on two inputs this
  // same function only writes ABOVE at the closure-publish site --
  // construction_closure_count (host member) and construction_closure_bytes
  // (H2D memcpy). Launched before them it saw count==0 and returned -- dead
  // by construction. Measured resident_plan_question_gap rows=0 on a corpus
  // with 36 question->answer adjacencies, whose single initial_exposure=true
  // pass was exactly the dead one.
  // Diagnostic readback: the distribution of `differences` over every evaluated
  // question/answer triple pair. Bin 1 is the only value that can vote; an
  // all-zero row means no pair was evaluated at all, and a distribution with
  // mass only at 2+ means the pairs exist but never differ in exactly one
  // coordinate. Host-side fprintf to stderr only -- DEVICE printf writes to
  // stdout, which is the duplex motor frame channel, and hangs the harness.
  if (std::getenv("BCC32_CLOSURE_DIAG") != nullptr &&
      state.question_gap_vote_histogram.get() != nullptr) {
    std::uint32_t gap_vote_histogram[8] = {};
    cuda_require(cudaMemcpy(gap_vote_histogram,
                            state.question_gap_vote_histogram.get(),
                            sizeof(gap_vote_histogram), cudaMemcpyDeviceToHost),
                 "read back question-gap vote histogram");
    std::fprintf(stderr,
                 "gap_vote_histogram d0=%u d1=%u d2=%u d3=%u d4=%u d5=%u "
                 "d6=%u d7=%u\n",
                 gap_vote_histogram[0], gap_vote_histogram[1],
                 gap_vote_histogram[2], gap_vote_histogram[3],
                 gap_vote_histogram[4], gap_vote_histogram[5],
                 gap_vote_histogram[6], gap_vote_histogram[7]);
  }
  // Companion diagnostic readback: WHICH coordinate differed, counted once per
  // differing coordinate of every evaluated pair (so the columns sum to the
  // difference-weighted total of the row above, not to its pair count). Same
  // host-side-fprintf-only rule as above.
  if (std::getenv("BCC32_CLOSURE_DIAG") != nullptr &&
      state.question_gap_coordinate_histogram.get() != nullptr) {
    std::uint32_t gap_coordinate_histogram[construction::kRelationFieldCount] =
        {};
    cuda_require(cudaMemcpy(gap_coordinate_histogram,
                            state.question_gap_coordinate_histogram.get(),
                            sizeof(gap_coordinate_histogram),
                            cudaMemcpyDeviceToHost),
                 "read back question-gap coordinate histogram");
    std::fprintf(stderr, "gap_coordinate_histogram c0=%u c1=%u c2=%u c3=%u\n",
                 gap_coordinate_histogram[0], gap_coordinate_histogram[1],
                 gap_coordinate_histogram[2], gap_coordinate_histogram[3]);
  }
  // Long sentences can exceed the bounded surface-slot capacity even though
  // their resident A/K/B matter is valid.  Retain the already-discovered
  // local relation frames as abstract surface witnesses so the ordered tissue
  // can hand a fact to realization without replaying its source sentence.
  DeviceArray<construction::RelationConstructionLearningReceipt>
      relation_construction_receipt(1u);
  cuda_require(cudaMemset(relation_construction_receipt.get(), 0,
                          relation_construction_receipt.bytes()),
               "reset resident relation construction receipt");
  const bool trace_relation_constructions =
      std::getenv("BCC32_RESIDENT_PLAN_DIAG") != nullptr;
  construction::learn_relation_constructions_kernel<<<
      (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
      sequence, sequence_count, segment_ids, state.construction_roles.get(),
      state.construction_closed_class_mask.get(), state.unit_lengths.get(),
      state.construction_tokens.get(), state.construction_lengths.get(),
      state.construction_slot_counts.get(), state.construction_supports.get(),
      state.construction_slot_units.get(), state.construction_slot_masses.get(),
      state.construction_slot_totals.get(), state.construction_slot_overflow.get(),
      state.construction_hash_slots.get(), state.construction_store_count.get(),
      state.proposition_ordered_evidence_revision.get(),
       state.construction_evidence_revision.get(),
       state.construction_origin_revision.get(),
       state.witnessed_relation_events.get(),
       state.witnessed_relation_event_cursor.get(),
       state.witnessed_relation_constructions.get(),
       state.witnessed_relation_surface_units.get(),
       state.witnessed_relation_surface_counts.get(),
       construction_allocation_capacity,
       trace_relation_constructions ? relation_construction_receipt.get()
                                    : nullptr);
  cuda_require(cudaGetLastError(), "learn resident relation constructions");
  if (state.proposition_ordered_construction.get() != nullptr) {
    ordered_relation::invalidate_current_ordered_construction_links_kernel<<<
        (state.proposition_ordered_bindings.size() + kBlock - 1u) / kBlock,
        kBlock>>>(
        state.proposition_ordered_bindings.get(),
        static_cast<std::uint32_t>(state.proposition_ordered_bindings.size()),
        state.proposition_ordered_evidence_revision.get(),
        state.proposition_ordered_construction.get());
    cuda_require(cudaGetLastError(),
                 "invalidate current ordered construction witnesses");
    ordered_relation::bridge_exact_event_constructions_kernel<<<
        (construction::kWitnessedRelationEventCap + kBlock - 1u) / kBlock,
        kBlock>>>(
        resident_proposition_tissue_view(state),
        state.witnessed_relation_events.get(),
        state.witnessed_relation_event_cursor.get(),
        state.witnessed_relation_constructions.get(),
        state.construction_store_count.get(), construction::kConstructionCap,
        resident_surface_population_view(state),
        state.proposition_ordered_construction.get());
    cuda_require(cudaGetLastError(), "bridge exact event construction witnesses");
  }
  // The unconditional synchronous cudaMemcpy below (and, when tracing is on,
  // the one inside the branch immediately above it) already blocks on the
  // legacy default stream, making an explicit cudaDeviceSynchronize() here
  // redundant (0X1-284).
  if (trace_relation_constructions) {
    construction::RelationConstructionLearningReceipt receipt{};
    cuda_require(cudaMemcpy(&receipt, relation_construction_receipt.get(),
                            sizeof(receipt), cudaMemcpyDeviceToHost),
                 "read resident relation construction receipt");
    std::fprintf(
        stderr,
        "resident_plan_relation_construction_learning attempted=%u learned=%u missing=%u linked=%u\n",
        receipt.attempted_events, receipt.learned_constructions,
        receipt.missing_constructions, receipt.linked_events);
  }
  std::uint32_t learned = 0u;
  cuda_require(cudaMemcpy(&learned, state.construction_store_count.get(),
                          sizeof(learned), cudaMemcpyDeviceToHost),
               "read resident construction extent");
  state.construction_count_host =
      std::min<std::uint32_t>(learned, construction::kConstructionCap);
  std::uint32_t construction_max_support = 0u;
  std::uint32_t construction_recurrent = 0u;
  std::uint32_t ordered_witness_links = 0u;
  if (std::getenv("BCC32_RESIDENT_PLAN_DIAG") != nullptr &&
      state.construction_count_host != 0u) {
    std::vector<std::uint32_t> supports(state.construction_count_host);
    cuda_require(cudaMemcpy(supports.data(), state.construction_supports.get(),
                            supports.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read resident construction support diagnostics");
    for (const std::uint32_t support : supports) {
      construction_max_support = std::max(construction_max_support, support);
      if (support >= construction::kConstructionMinRoleEvidence)
        ++construction_recurrent;
    }
    std::vector<std::uint32_t> links(state.proposition_ordered_construction.size());
    cuda_require(cudaMemcpy(links.data(), state.proposition_ordered_construction.get(),
                            links.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read ordered construction witness diagnostics");
    for (const std::uint32_t link : links)
      ordered_witness_links += link != construction::kNoConstruction;
  }
  std::uint32_t closure_probe[4u] = {};
  cuda_require(cudaMemcpy(closure_probe, state.construction_closure_bytes.get(),
                          sizeof(closure_probe), cudaMemcpyDeviceToHost),
               "read resident closure bytes for construction diagnostics");
  std::vector<roles::MutableStructuralRole> host_roles(state.unit_count);
  cuda_require(cudaMemcpy(host_roles.data(), state.construction_roles.get(),
                          host_roles.size() * sizeof(roles::MutableStructuralRole),
                          cudaMemcpyDeviceToHost),
               "read resident construction roles for diagnostics");
  std::vector<std::uint32_t> vitality_again(state.unit_count);
  cuda_require(cudaMemcpy(vitality_again.data(), state.unit_vitality.get(),
                          vitality_again.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "read resident vitality for diagnostics");
  std::uint32_t closed_class_units = 0u;
  std::uint32_t confident_units = 0u;
  for (std::uint32_t unit = 0u; unit < state.unit_count; ++unit) {
    if (host_roles[unit].confidence != 0u) ++confident_units;
    if (host_roles[unit].confidence != 0u &&
        host_roles[unit].role < roles::kStructuralRoleCount &&
        vitality_again[unit] >= state.construction_func_threshold &&
        role_population[host_roles[unit].role] != 0u &&
        role_population[host_roles[unit].role] <=
            state.construction_role_population_cutoff)
      ++closed_class_units;
  }
  std::fprintf(stderr,
               "construction_store=%u func_threshold=%u episodes=%u"
               " closure_bytes=%u,%u,%u,%u closure_count=%u"
               " role_population_cutoff=%u closed_class_units=%u"
               " confident_units=%u support_max=%u recurrent=%u witness_links=%u\n",
               state.construction_count_host, state.construction_func_threshold,
               episode_count, closure_probe[0], closure_probe[1],
               closure_probe[2], closure_probe[3],
               state.construction_closure_count,
               state.construction_role_population_cutoff, closed_class_units,
               confident_units, construction_max_support, construction_recurrent,
               ordered_witness_links);
}
