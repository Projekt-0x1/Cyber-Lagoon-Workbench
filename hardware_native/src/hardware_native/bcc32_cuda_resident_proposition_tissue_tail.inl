// One sensory path handles both observation and intervention. `efferent_contact`
// says whether the adult's body produced a target-directed action; it does not
// name a relation. For intervention contacts, polarity is the physical direction
// of the action and outcome_present is the later sensory consequence. Concordant
// restore/presence and block/absence contacts strengthen the binding, while the
// opposite outcomes write counterevidence.
__device__ inline bool assimilate_experience(
    TissueView tissue, SparsePopulationView earlier,
    SparsePopulationView later_reference, SparsePopulationView context,
    std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present) {
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr ||
      tissue.synapse_capacity == 0u ||
      !valid_population(earlier, tissue.cell_capacity) ||
      !valid_population(later_reference, tissue.cell_capacity) ||
      !valid_population(context, tissue.cell_capacity) ||
      (efferent_contact != 0u && efferent_polarity == 0))
    return false;
  const bool intervention = efferent_contact != 0u;
  const bool concordant = !intervention || ((efferent_polarity > 0) == (outcome_present != 0u));
  const std::uint64_t first_evidence_mass =
      1u + earlier.count + 1u + context.count +
      (intervention ? kInterventionQuantum : kObservationQuantum);
  bool accepted = false;
  for (std::uint32_t source = 0u; source < earlier.count; ++source) {
    for (std::uint32_t target = 0u; target < later_reference.count; ++target) {
      SparseBindingSynapse* synapse =
          find_or_claim_synapse(tissue, earlier.cells[source], later_reference.cells[target],
                                first_evidence_mass);
      if (synapse == nullptr)
        continue;
      PopulationCohortEvidence* cohort =
          find_or_claim_cohort(tissue, synapse, earlier);
      if (cohort == nullptr)
        continue;
      const std::uint64_t observation_before = cohort->observational_support;
      const std::uint64_t intervention_before = cohort->intervention_support;
      const std::uint64_t counter_before = cohort->counterevidence;
      retain_qualifying_context(tissue.scalars, cohort, context);
      if (!intervention) {
        move_support_mass(tissue.scalars, cohort, false, kObservationQuantum);
      } else if (concordant) {
        move_support_mass(tissue.scalars, cohort, true, kInterventionQuantum);
      } else {
        move_counter_mass(tissue.scalars, cohort, kInterventionQuantum);
      }
      const bool synapse_credited = observation_before != cohort->observational_support ||
                                    intervention_before != cohort->intervention_support ||
                                    counter_before != cohort->counterevidence;
      // Mark set at the same instant the credit above lands, so a contact and its own
      // mark-and-sweep record can never come apart. This is the tissue's own memory of
      // "contacted since the last drive" -- apply_unreinforced_support_drain reads and
      // clears it; nothing else does.
      if (synapse_credited)
        synapse->contacted_since_drain = 1u;
      accepted = accepted || synapse_credited;
    }
  }
  if (!accepted)
    return false;
  ++tissue.scalars->revision;
  tissue.scalars->accepted_observations += !intervention;
  tissue.scalars->accepted_interventions += intervention && concordant;
  tissue.scalars->accepted_counterevidence += intervention && !concordant;
  return true;
}

__device__ inline bool assimilate_experience_warp(
    TissueView tissue, SparsePopulationView earlier,
    SparsePopulationView later_reference, SparsePopulationView context,
    std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present, std::uint32_t repetitions = 1u) {
  const unsigned active = __activemask();
  const std::uint32_t lane = threadIdx.x & 31u;
  const bool valid = tissue.synapses != nullptr && tissue.scalars != nullptr &&
                     tissue.cohorts != nullptr &&
                     tissue.synapse_capacity != 0u &&
                     valid_population(earlier, tissue.cell_capacity) &&
                     valid_population(later_reference, tissue.cell_capacity) &&
                     valid_population(context, tissue.cell_capacity) &&
                     (efferent_contact == 0u || efferent_polarity != 0);
  if (!valid)
    return false;
  const bool intervention = efferent_contact != 0u;
  const bool concordant =
      !intervention || ((efferent_polarity > 0) == (outcome_present != 0u));
  const std::uint64_t first_evidence_mass =
      1u + earlier.count + 1u + context.count +
      (intervention ? kInterventionQuantum : kObservationQuantum);
  std::uint32_t accepted_repetitions = 0u;
  for (std::uint32_t source = 0u; source < earlier.count; ++source) {
    for (std::uint32_t target = 0u; target < later_reference.count; ++target) {
      SparseBindingSynapse* synapse = find_or_claim_synapse_warp(
          tissue, earlier.cells[source], later_reference.cells[target],
          first_evidence_mass);
      if (lane == 0u && synapse != nullptr) {
        PopulationCohortEvidence* cohort =
            find_or_claim_cohort(tissue, synapse, earlier);
        if (cohort != nullptr) {
          const std::uint64_t observation_before = cohort->observational_support;
          const std::uint64_t intervention_before = cohort->intervention_support;
          const std::uint64_t counter_before = cohort->counterevidence;
          retain_qualifying_context(tissue.scalars, cohort, context);
          std::uint32_t moved_repetitions = 0u;
          if (!intervention)
            moved_repetitions = move_support_mass_repeated(
                tissue.scalars, cohort, false, kObservationQuantum, repetitions);
          else if (concordant)
            moved_repetitions = move_support_mass_repeated(
                tissue.scalars, cohort, true, kInterventionQuantum, repetitions);
          else
            moved_repetitions = move_counter_mass_repeated(
                tissue.scalars, cohort, kInterventionQuantum, repetitions);
          if (observation_before != cohort->observational_support ||
              intervention_before != cohort->intervention_support ||
              counter_before != cohort->counterevidence) {
            // Same mark, same instant-of-credit rule as the scalar path above.
            synapse->contacted_since_drain = 1u;
            accepted_repetitions =
                accepted_repetitions > moved_repetitions
                    ? accepted_repetitions
                    : moved_repetitions;
          }
        }
      }
      __syncwarp(active);
    }
  }
  accepted_repetitions = __shfl_sync(active, accepted_repetitions, 0);
  if (lane == 0u && accepted_repetitions != 0u) {
    tissue.scalars->revision += accepted_repetitions;
    tissue.scalars->accepted_observations +=
        !intervention ? accepted_repetitions : 0u;
    tissue.scalars->accepted_interventions +=
        intervention && concordant ? accepted_repetitions : 0u;
    tissue.scalars->accepted_counterevidence +=
        intervention && !concordant ? accepted_repetitions : 0u;
  }
  __syncwarp(active);
  return accepted_repetitions != 0u;
}


__device__ inline OrderedRoleBindingEvidence* find_ordered_binding(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, OrderedRoleBindingEvidence** empty_slot = nullptr,
    OrderedRoleUnitIdentity identity = {}) {
  if (empty_slot != nullptr)
    *empty_slot = nullptr;
  if (tissue.ordered_bindings == nullptr || tissue.ordered_binding_capacity == 0u)
    return nullptr;
  const std::uint32_t begin =
      (ordered_identity_present(identity)
           ? ordered_identity_hash(identity)
           : ordered_binding_hash(agent, predicate, patient)) %
      tissue.ordered_binding_capacity;
  const std::uint32_t probe_limit =
      tissue.ordered_binding_capacity < kMaximumOrderedBindingProbe
          ? tissue.ordered_binding_capacity
          : kMaximumOrderedBindingProbe;
  for (std::uint32_t probe = 0u; probe < probe_limit; ++probe) {
    OrderedRoleBindingEvidence* candidate =
        tissue.ordered_bindings + (begin + probe) % tissue.ordered_binding_capacity;
    if (candidate->claimed == 0u) {
      if (empty_slot != nullptr && *empty_slot == nullptr)
        *empty_slot = candidate;
      continue;
    }
    if (same_ordered_binding(*candidate, agent, predicate, patient) &&
        same_ordered_identity(*candidate, identity))
      return candidate;
  }
  return nullptr;
}

// Retain one complete ordered proposition. A monotonically increasing physical
// contact revision supplies episode provenance; replaying the same retained
// revision is idempotent. No local adjacency, digest, or role-category
// substitution can create or authorize the record.
__device__ inline OrderedBindingAssimilationResult assimilate_ordered_binding_detailed(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, SparsePopulationView context,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    OrderedRoleUnitIdentity identity = {}) {
  OrderedBindingAssimilationResult result{};
  result.projected_bytes =
      static_cast<std::uint64_t>(tissue.ordered_binding_capacity) *
      sizeof(OrderedRoleBindingEvidence);
  if (tissue.scalars != nullptr)
    result.tissue_revision = tissue.scalars->revision;
  if (tissue.scalars == nullptr || tissue.ordered_bindings == nullptr ||
      tissue.ordered_binding_capacity == 0u || evidence_revision == 0u ||
      !valid_distinct_population(agent, tissue.cell_capacity) ||
      !valid_distinct_population(predicate, tissue.cell_capacity) ||
      !valid_distinct_population(patient, tissue.cell_capacity) ||
      !valid_distinct_population(context, tissue.cell_capacity) ||
      !ordered_identity_valid(identity) ||
      (efferent_contact != 0u && efferent_polarity == 0)) {
    if (tissue.ordered_binding_admission != nullptr)
      ++tissue.ordered_binding_admission->invalid_attempts;
    return result;
  }

  OrderedRoleBindingEvidence* empty = nullptr;
  OrderedRoleBindingEvidence* binding =
      find_ordered_binding(tissue, agent, predicate, patient, &empty, identity);
  const bool is_new = binding == nullptr;
  if (is_new) {
    if (empty == nullptr) {
      result.status = OrderedBindingAssimilationStatus::capacity_overflow;
      if (tissue.ordered_binding_admission != nullptr) {
        ++tissue.ordered_binding_admission->capacity_rejections;
        if (tissue.scalars->free_mass != 0u) {
          --tissue.scalars->free_mass;
          ++tissue.ordered_binding_admission->overflow_mass;
          ++tissue.scalars->revision;
          result.tissue_revision = tissue.scalars->revision;
        }
      }
      return result;
    }
    binding = empty;
  } else if (evidence_revision <= binding->last_evidence_revision) {
    result.status = OrderedBindingAssimilationStatus::replay;
    result.binding_index =
        static_cast<std::uint32_t>(binding - tissue.ordered_bindings);
    if (tissue.ordered_binding_admission != nullptr)
      ++tissue.ordered_binding_admission->replays;
    return result;
  }

  const bool intervention = efferent_contact != 0u;
  // For passive observation, proposition presence is the evidence and the
  // outcome bit is deliberately ignored. Negative/corrective evidence enters
  // through an efferent action whose later consequence disagrees with its
  // physical polarity; it is never inferred from absent punctuation or text.
  const bool concordant =
      !intervention || ((efferent_polarity > 0) == (outcome_present != 0u));
  const bool retain_context =
      (is_new || ordered_binding_context_slot(*binding, context) < 0) &&
      (is_new || binding->qualifying_context_count < kMinimumIndependentContexts);
  const std::uint64_t structure_cost =
      is_new ? 1u + agent.count + predicate.count + patient.count +
                   identity.counts[0] + identity.counts[1] + identity.counts[2]
             : 0u;
  const std::uint64_t context_cost = retain_context ? 1u + context.count : 0u;
  std::uint64_t evidence_cost = 0u;
  if (!intervention) {
    evidence_cost = kObservationQuantum;
    if (retain_context)
      evidence_cost += kObservationQuantum;
  } else {
    evidence_cost = kInterventionQuantum;
  }
  const std::uint64_t required_mass = structure_cost + context_cost + evidence_cost;
  result.required_mass = required_mass;
  if (tissue.scalars->free_mass < required_mass) {
    result.status = OrderedBindingAssimilationStatus::insufficient_mass;
    if (tissue.ordered_binding_admission != nullptr)
      ++tissue.ordered_binding_admission->mass_rejections;
    return result;
  }

  if (is_new) {
    *binding = OrderedRoleBindingEvidence{};
    binding->claimed = 1u;
    const SparsePopulationView roles[kOrderedBindingRoleCount] = {
        agent, predicate, patient};
    for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role) {
      binding->role_counts[role] = roles[role].count;
      binding->role_unit_counts[role] = identity.counts[role];
      binding->role_structure_mass[role] =
          roles[role].count + identity.counts[role];
      for (std::uint32_t index = 0u; index < identity.counts[role]; ++index)
        binding->role_units[role][index] = identity.units[role][index];
      for (std::uint32_t cell = 0u; cell < roles[role].count; ++cell)
        binding->role_cells[role][cell] = roles[role].cells[cell];
    }
    binding->structure_mass = structure_cost;
    tissue.scalars->free_mass -= structure_cost;
  }
  if (retain_context) {
    const std::uint32_t slot = binding->qualifying_context_count++;
    binding->qualifying_context_counts[slot] = context.count;
    binding->qualifying_episode_revisions[slot] = evidence_revision;
    for (std::uint32_t cell = 0u; cell < context.count; ++cell)
      binding->qualifying_context_cells[slot][cell] = context.cells[cell];
    binding->context_mass += context_cost;
    tissue.scalars->free_mass -= context_cost;
  }
  if (!intervention) {
    binding->episodic_observation_mass += kObservationQuantum;
    tissue.scalars->free_mass -= kObservationQuantum;
    if (retain_context) {
      binding->generalized_observation_mass += kObservationQuantum;
      tissue.scalars->free_mass -= kObservationQuantum;
    }
    ++tissue.scalars->accepted_observations;
  } else if (concordant) {
    binding->intervention_support += kInterventionQuantum;
    tissue.scalars->free_mass -= kInterventionQuantum;
    ++tissue.scalars->accepted_interventions;
  } else {
    binding->counterevidence += kInterventionQuantum;
    tissue.scalars->free_mass -= kInterventionQuantum;
    ++tissue.scalars->accepted_counterevidence;
  }
  binding->last_evidence_revision = evidence_revision;
  ++tissue.scalars->revision;
  if (tissue.ordered_binding_admission != nullptr)
    ++tissue.ordered_binding_admission->accepted;
  result.status = OrderedBindingAssimilationStatus::accepted;
  result.binding_index =
      static_cast<std::uint32_t>(binding - tissue.ordered_bindings);
  result.tissue_revision = tissue.scalars->revision;
  return result;
}

__device__ inline bool assimilate_ordered_binding(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, SparsePopulationView context,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present) {
  const auto result = assimilate_ordered_binding_detailed(
      tissue, agent, predicate, patient, context, evidence_revision,
      efferent_contact, efferent_polarity, outcome_present);
  return result.status == OrderedBindingAssimilationStatus::accepted ||
         result.status == OrderedBindingAssimilationStatus::replay;
}

__device__ inline bool assimilate_ordered_binding_warp(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, SparsePopulationView context,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present) {
  const unsigned active = __activemask();
  bool accepted = false;
  if ((threadIdx.x & 31u) == 0u)
    accepted = assimilate_ordered_binding(
        tissue, agent, predicate, patient, context, evidence_revision,
        efferent_contact, efferent_polarity, outcome_present);
  accepted = __shfl_sync(active, accepted, 0u) != 0;
  __syncwarp(active);
  return accepted;
}

__device__ inline OrderedBindingLookupResult collect_topic_ordered_bindings(
    TissueView tissue, SparsePopulationView topic,
    OrderedBindingQualification qualification, std::uint32_t* output_indices,
    std::uint32_t output_capacity) {
  OrderedBindingLookupResult result{};
  if (tissue.scalars != nullptr)
    result.tissue_revision = tissue.scalars->revision;
  if (tissue.ordered_bindings == nullptr || output_indices == nullptr ||
      output_capacity == 0u || !valid_distinct_population(topic, tissue.cell_capacity))
    return result;
  std::uint32_t qualified = 0u;
  for (std::uint32_t index = 0u; index < tissue.ordered_binding_capacity; ++index) {
    const OrderedRoleBindingEvidence& binding = tissue.ordered_bindings[index];
    if (binding.claimed == 0u)
      continue;
    const bool topic_match =
        exact_population_equals(binding.role_cells[0], binding.role_counts[0], topic) ||
        exact_population_equals(binding.role_cells[2], binding.role_counts[2], topic);
    if (!topic_match)
      continue;
    ++result.exact_topic_matches;
    qualified += ordered_binding_qualified(binding, qualification,
                                            tissue.cell_capacity);
  }
  if (qualified == 0u)
    return result;
  if (qualified > output_capacity) {
    result.overflow = 1u;
    return result;
  }
  for (std::uint32_t index = 0u; index < tissue.ordered_binding_capacity; ++index) {
    const OrderedRoleBindingEvidence& binding = tissue.ordered_bindings[index];
    const bool topic_match = binding.claimed != 0u &&
        (exact_population_equals(binding.role_cells[0], binding.role_counts[0], topic) ||
         exact_population_equals(binding.role_cells[2], binding.role_counts[2], topic));
    if (topic_match && ordered_binding_qualified(binding, qualification,
                                                 tissue.cell_capacity))
      output_indices[result.output_count++] = index;
  }
  result.ready = result.output_count != 0u;
  return result;
}

__device__ inline bool lesion_ordered_binding_role_cell(
    TissueView tissue, std::uint32_t binding_index, std::uint32_t role,
    std::uint64_t* escrow_mass) {
  if (tissue.scalars == nullptr || tissue.ordered_bindings == nullptr ||
      escrow_mass == nullptr || binding_index >= tissue.ordered_binding_capacity ||
      role >= kOrderedBindingRoleCount)
    return false;
  OrderedRoleBindingEvidence& binding = tissue.ordered_bindings[binding_index];
  if (binding.claimed == 0u || binding.role_structure_mass[role] == 0u ||
      binding.structure_mass == 0u)
    return false;
  --binding.role_structure_mass[role];
  --binding.structure_mass;
  ++*escrow_mass;
  ++tissue.scalars->lesion_revision;
  ++tissue.scalars->revision;
  return true;
}

__device__ inline bool restore_ordered_binding_role_cell(
    TissueView tissue, std::uint32_t binding_index, std::uint32_t role,
    std::uint64_t* escrow_mass) {
  if (tissue.scalars == nullptr || tissue.ordered_bindings == nullptr ||
      escrow_mass == nullptr || *escrow_mass == 0u ||
      binding_index >= tissue.ordered_binding_capacity ||
      role >= kOrderedBindingRoleCount)
    return false;
  OrderedRoleBindingEvidence& binding = tissue.ordered_bindings[binding_index];
  if (binding.claimed == 0u ||
      binding.role_structure_mass[role] >= binding.role_counts[role])
    return false;
  ++binding.role_structure_mass[role];
  ++binding.structure_mass;
  --*escrow_mass;
  ++tissue.scalars->lesion_revision;
  ++tissue.scalars->revision;
  return true;
}

__global__ void assimilate_ordered_binding_kernel(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, SparsePopulationView context,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    std::uint32_t* accepted) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const bool result = assimilate_ordered_binding(
      tissue, agent, predicate, patient, context, evidence_revision,
      efferent_contact, efferent_polarity, outcome_present);
  if (accepted != nullptr)
    *accepted = result ? 1u : 0u;
}

__global__ void assimilate_ordered_binding_detailed_kernel(
    TissueView tissue, SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient, SparsePopulationView context,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    OrderedBindingAssimilationResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = assimilate_ordered_binding_detailed(
      tissue, agent, predicate, patient, context, evidence_revision,
      efferent_contact, efferent_polarity, outcome_present);
}

__global__ void collect_topic_ordered_bindings_kernel(
    TissueView tissue, SparsePopulationView topic,
    OrderedBindingQualification qualification, std::uint32_t* output_indices,
    std::uint32_t output_capacity, OrderedBindingLookupResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = collect_topic_ordered_bindings(
      tissue, topic, qualification, output_indices, output_capacity);
}

__global__ void assimilate_experience_kernel(TissueView tissue, SparsePopulationView earlier,
                                             SparsePopulationView later_reference,
                                             SparsePopulationView context,
                                             std::uint32_t efferent_contact,
                                             std::int32_t efferent_polarity,
                                             std::uint32_t outcome_present) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  assimilate_experience(tissue, earlier, later_reference, context,
                        efferent_contact, efferent_polarity, outcome_present);
}

// Candidate sensory encoder for continuous raw episodes. Sentence/episode
// boundaries are learned upstream; this organ assigns no word class or relation
// name. Each episode contributes weak support between overlapping populations
// at its two temporal edges. Body consequence evidence uses this same encoder
// with efferent_contact and physical outcome metadata set.
__device__ inline void assimilate_candidate_span(
    TissueView tissue, const std::uint32_t* sequence, std::uint32_t begin,
    std::uint32_t end, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    const std::uint32_t* unit_populations, std::uint32_t unit_count,
    std::uint32_t unit_population_width) {
  const std::uint32_t length = end - begin;
  // Odd episodes have a real shared middle population.  Retaining it on both
  // temporal sides lets the complete observed prefix condition the later
  // edge; floor division silently erased that relation-bearing population.
  std::uint32_t edge_units = (length + 1u) / 2u;
  if (edge_units > 4u)
    edge_units = 4u;
  std::uint32_t sampled_units = length;
  if (sampled_units > kMaximumPopulationCells / 2u)
    sampled_units = kMaximumPopulationCells / 2u;
  std::uint32_t context_cells[kMaximumPopulationCells]{};
  std::uint32_t context_count = 0u;
  for (std::uint32_t slot = 0u; slot < sampled_units; ++slot) {
    const std::uint32_t offset = sampled_units == 1u
        ? 0u
        : slot * (length - 1u) / (sampled_units - 1u);
    const std::uint32_t unit = sequence[begin + offset];
    if (unit >= unit_count)
      return;
    const std::uint32_t* cells =
        unit_populations + static_cast<std::size_t>(unit) * unit_population_width;
    const std::uint32_t sampled_cell_count = unit_population_width > 1u ? 2u : 1u;
    for (std::uint32_t cell_index = 0u; cell_index < sampled_cell_count; ++cell_index) {
      bool present = false;
      for (std::uint32_t existing = 0u; existing < context_count; ++existing)
        present = present || context_cells[existing] == cells[cell_index];
      if (!present)
        context_cells[context_count++] = cells[cell_index];
    }
  }
  for (std::uint32_t source = 0u; source < edge_units; ++source) {
    const std::uint32_t source_unit = sequence[begin + source];
    if (source_unit >= unit_count)
      return;
    const auto earlier = SparsePopulationView{
        unit_populations +
            static_cast<std::size_t>(source_unit) * unit_population_width,
        unit_population_width};
    for (std::uint32_t target = 0u; target < edge_units; ++target) {
      const std::uint32_t target_unit = sequence[end - edge_units + target];
      if (target_unit >= unit_count)
        return;
      const auto later = SparsePopulationView{
          unit_populations +
              static_cast<std::size_t>(target_unit) * unit_population_width,
          unit_population_width};
      assimilate_experience(
          tissue, earlier, later,
          SparsePopulationView{context_cells, context_count}, efferent_contact,
          efferent_polarity, outcome_present);
    }
  }
}

__device__ inline void assimilate_candidate_span_warp(
    TissueView tissue, const std::uint32_t* sequence, std::uint32_t begin,
    std::uint32_t end, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    const std::uint32_t* unit_populations, std::uint32_t unit_count,
    std::uint32_t unit_population_width, std::uint32_t repetitions = 1u) {
  const std::uint32_t length = end - begin;
  std::uint32_t edge_units = (length + 1u) / 2u;
  if (edge_units > 4u)
    edge_units = 4u;
  std::uint32_t sampled_units = length;
  if (sampled_units > kMaximumPopulationCells / 2u)
    sampled_units = kMaximumPopulationCells / 2u;
  std::uint32_t context_cells[kMaximumPopulationCells]{};
  std::uint32_t context_count = 0u;
  for (std::uint32_t slot = 0u; slot < sampled_units; ++slot) {
    const std::uint32_t offset = sampled_units == 1u
                                     ? 0u
                                     : slot * (length - 1u) / (sampled_units - 1u);
    const std::uint32_t unit = sequence[begin + offset];
    if (unit >= unit_count)
      return;
    const std::uint32_t* cells =
        unit_populations + static_cast<std::size_t>(unit) * unit_population_width;
    const std::uint32_t sampled_cell_count = unit_population_width > 1u ? 2u : 1u;
    for (std::uint32_t cell_index = 0u; cell_index < sampled_cell_count; ++cell_index) {
      bool present = false;
      for (std::uint32_t existing = 0u; existing < context_count; ++existing)
        present = present || context_cells[existing] == cells[cell_index];
      if (!present)
        context_cells[context_count++] = cells[cell_index];
    }
  }
  for (std::uint32_t source = 0u; source < edge_units; ++source) {
    const std::uint32_t source_unit = sequence[begin + source];
    if (source_unit >= unit_count)
      return;
    const auto earlier = SparsePopulationView{
        unit_populations +
            static_cast<std::size_t>(source_unit) * unit_population_width,
        unit_population_width};
    for (std::uint32_t target = 0u; target < edge_units; ++target) {
      const std::uint32_t target_unit = sequence[end - edge_units + target];
      if (target_unit >= unit_count)
        return;
      const auto later = SparsePopulationView{
          unit_populations +
              static_cast<std::size_t>(target_unit) * unit_population_width,
          unit_population_width};
      assimilate_experience_warp(
          tissue, earlier, later, SparsePopulationView{context_cells, context_count},
          efferent_contact, efferent_polarity, outcome_present, repetitions);
    }
  }
}

__global__ void assimilate_candidate_episodes_kernel(
    TissueView tissue, const std::uint32_t* sequence,
    const std::uint32_t* segment_ids, std::uint32_t sequence_count,
    std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present, const std::uint32_t* unit_populations,
    std::uint32_t unit_count, std::uint32_t unit_population_width) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || sequence == nullptr ||
      segment_ids == nullptr || sequence_count < 2u ||
      unit_populations == nullptr || unit_population_width == 0u ||
      unit_population_width > kMaximumPopulationCells)
    return;
  std::uint32_t encoded_episodes = 0u;
  std::uint32_t begin = 0u;
  while (begin < sequence_count) {
    std::uint32_t end = begin + 1u;
    while (end < sequence_count && segment_ids[end] == segment_ids[begin])
      ++end;
    if (end - begin >= 2u) {
      assimilate_candidate_span(tissue, sequence, begin, end,
                                 efferent_contact, efferent_polarity,
                                 outcome_present, unit_populations, unit_count,
                                 unit_population_width);
      ++encoded_episodes;
    }
    begin = end;
  }
  // An immature closure field can transiently mark every unit in a short
  // contact as a separate segment. Preserve that bounded contact as one
  // candidate episode. A large, already segmented stream is not one
  // proposition: folding megabytes into a single span both invents
  // cross-episode bindings and turns this local learner into an O(n) serial
  // source walk.
  constexpr std::uint32_t kMaximumUnsegmentedFallbackUnits = 64u;
  if (encoded_episodes == 0u &&
      sequence_count <= kMaximumUnsegmentedFallbackUnits)
    assimilate_candidate_span(tissue, sequence, 0u, sequence_count,
                              efferent_contact, efferent_polarity,
                              outcome_present, unit_populations, unit_count,
                              unit_population_width);
}

// Optimized production entry. The complete warp advances through the same
// episode/source/target ordinals as the scalar reference; only the bounded
// synapse probe loads are parallel.
__global__ void assimilate_candidate_episodes_warp_kernel(
    TissueView tissue, const std::uint32_t* sequence,
    const std::uint32_t* segment_ids, std::uint32_t sequence_count,
    std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present, const std::uint32_t* unit_populations,
    std::uint32_t unit_count, std::uint32_t unit_population_width) {
  if (blockIdx.x != 0u || threadIdx.x >= 32u || blockDim.x < 32u ||
      sequence == nullptr || segment_ids == nullptr || sequence_count < 2u ||
      unit_populations == nullptr || unit_population_width == 0u ||
      unit_population_width > kMaximumPopulationCells)
    return;
  constexpr std::uint32_t kRepeatedSpanCache = 64u;
  __shared__ std::uint32_t cached_begin[kRepeatedSpanCache];
  __shared__ std::uint32_t cached_end[kRepeatedSpanCache];
  __shared__ std::uint32_t cached_repetitions[kRepeatedSpanCache];
  __shared__ std::uint32_t cached_count;
  if (threadIdx.x == 0u)
    cached_count = 0u;
  __syncwarp();
  if (threadIdx.x == 0u) {
    std::uint32_t scan = 0u;
    while (scan < sequence_count) {
      std::uint32_t end = scan + 1u;
      while (end < sequence_count && segment_ids[end] == segment_ids[scan])
        ++end;
      if (end - scan >= 2u) {
        std::uint32_t match = cached_count;
        for (std::uint32_t candidate = 0u; candidate < cached_count; ++candidate) {
          const std::uint32_t candidate_length =
              cached_end[candidate] - cached_begin[candidate];
          if (candidate_length != end - scan)
            continue;
          bool same = true;
          for (std::uint32_t offset = 0u; offset < candidate_length; ++offset)
            same = same &&
                   sequence[cached_begin[candidate] + offset] ==
                       sequence[scan + offset];
          if (same) {
            match = candidate;
            break;
          }
        }
        if (match < cached_count) {
          ++cached_repetitions[match];
        } else if (cached_count < kRepeatedSpanCache) {
          cached_begin[cached_count] = scan;
          cached_end[cached_count] = end;
          cached_repetitions[cached_count] = 1u;
          ++cached_count;
        }
      }
      scan = end;
    }
  }
  __syncwarp();
  std::uint32_t encoded_episodes = 0u;
  std::uint32_t begin = 0u;
  while (begin < sequence_count) {
    std::uint32_t end = begin + 1u;
    while (end < sequence_count && segment_ids[end] == segment_ids[begin])
      ++end;
    if (end - begin >= 2u) {
      std::uint32_t repetitions = 1u;
      bool cached_representative = false;
      for (std::uint32_t candidate = 0u; candidate < cached_count; ++candidate) {
        if (cached_begin[candidate] == begin) {
          repetitions = cached_repetitions[candidate];
          cached_representative = true;
          break;
        }
        if (cached_end[candidate] - cached_begin[candidate] != end - begin)
          continue;
        bool same = true;
        for (std::uint32_t offset = 0u; offset < end - begin; ++offset)
          same = same &&
                 sequence[cached_begin[candidate] + offset] ==
                     sequence[begin + offset];
        if (same) {
          cached_representative = false;
          break;
        }
      }
      if (!cached_representative && cached_count < kRepeatedSpanCache) {
        begin = end;
        continue;
      }
      assimilate_candidate_span_warp(
          tissue, sequence, begin, end, efferent_contact, efferent_polarity,
          outcome_present, unit_populations, unit_count, unit_population_width,
          repetitions);
      ++encoded_episodes;
    }
    begin = end;
  }
  constexpr std::uint32_t kMaximumUnsegmentedFallbackUnits = 64u;
  if (encoded_episodes == 0u &&
      sequence_count <= kMaximumUnsegmentedFallbackUnits)
    assimilate_candidate_span_warp(
        tissue, sequence, 0u, sequence_count, efferent_contact,
        efferent_polarity, outcome_present, unit_populations, unit_count,
        unit_population_width);
}

__device__ inline bool causally_qualified(const PopulationCohortEvidence& cohort) {
  const std::uint64_t support = cohort.observational_support + cohort.intervention_support;
  return cohort.qualifying_context_count >= kMinimumIndependentContexts &&
         cohort.intervention_support >= kMinimumInterventionMass &&
         support > cohort.counterevidence;
}

// Reading can support a predictive discourse continuation without licensing
// an intervention-level causal claim.  The two policies deliberately share
// tissue but not their admission threshold: causal completion still requires
// action/outcome evidence, while discourse completion may use repeated,
// independently contextualized observations.
__device__ inline bool discourse_qualified(const PopulationCohortEvidence& cohort) {
  const std::uint64_t support =
      cohort.observational_support + cohort.intervention_support;
  // Discourse is graded recall, not a causal assertion.  A newly learned
  // episode with net-positive resident evidence must remain addressable; later
  // independent contexts strengthen its score instead of acting as a binary
  // existence gate.  Intervention-level readout remains strict in
  // causally_qualified above.
  return support > cohort.counterevidence;
}

__device__ inline bool qualified_for_completion(
    const PopulationCohortEvidence& cohort, CompletionPolicy policy) {
  return policy == CompletionPolicy::causal ? causally_qualified(cohort)
                                             : discourse_qualified(cohort);
}

struct SelectedCohortEvidence {
  std::uint64_t support = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint32_t overlap = 0u;
  std::uint32_t qualified = 0u;
  std::uint32_t ambiguous = 0u;
};

__device__ inline std::uint64_t net_evidence(std::uint64_t support,
                                             std::uint64_t counterevidence) {
  return support > counterevidence ? support - counterevidence : 0u;
}

// The table is open-addressed: relations that collide on a home bucket lie
// along ONE forward probe chain inside a window bounded by kMaximumSynapseProbe.
// A slot that once held a relation and was then vacated -- by a lesion, or by
// turnover returning a victim's mass -- is NOT the end of that chain.  Every
// relation behind the vacancy is still fully resident.
//
// This probe therefore never terminates on an unclaimed slot; it walks the
// whole bounded window and answers on exact relation identity alone.  A probe
// that stopped at the first unclaimed slot reported live, untouched relations
// as ABSENT whenever a hole opened ahead of them in a shared chain -- a false
// negative manufactured by the instrument, in exactly the lesion arms that read
// through this lookup.  Walking the window is what a deleted-slot marker would
// buy, without a marker: the window is bounded by construction, and no relation
// can ever be claimed outside its own window, so a full window scan is exact.
// Identity is compared exactly, so continuing past a hole can only ever find a
// relation that IS resident -- a relation that was never laid down still reads
// absent.
__device__ inline const SparseBindingSynapse* find_synapse_read_only(
    TissueView tissue, std::uint32_t source, std::uint32_t target) {
  const std::uint32_t begin = binding_hash(source, target) % tissue.synapse_capacity;
  const std::uint32_t probe_limit = tissue.synapse_capacity < kMaximumSynapseProbe
                                        ? tissue.synapse_capacity
                                        : kMaximumSynapseProbe;
  for (std::uint32_t probe = 0u; probe < probe_limit; ++probe) {
    const SparseBindingSynapse* synapse =
        tissue.synapses + (begin + probe) % tissue.synapse_capacity;
    if (synapse->claimed != 0u && synapse->source_cell == source &&
        synapse->target_cell == target)
      return synapse;
  }
  return nullptr;
}

// A per-call credit receipt reconstructed purely from before/after field deltas around ONE
// assimilate_experience call -- assimilate_experience itself is untouched, so every one of its
// other 47+ callers is unaffected. This is deliberately scoped to the REPEATED-CONTACT case on an
// ALREADY-CLAIMED synapse and cohort (exactly the material a plasticity-cycle already holds,
// claimed once and never re-claimed): it does NOT attempt to reverse a FIRST synapse/cohort claim
// (structure_mass allocation). credited stays false rather than mislabel a call this receipt
// cannot exactly invert -- the same discipline arm 4/arm 8 already apply to eviction and lesion.
struct ExperienceCreditReceipt {
  bool credited = false;
  std::uint32_t synapse_index = UINT32_MAX;
  std::uint32_t cohort_index = UINT32_MAX;
  bool moved_observational = false;
  bool moved_intervention = false;
  bool moved_counter = false;
  std::uint64_t moved = 0u;
  std::uint64_t cancelled_counter = 0u;       // observational/intervention path
  std::uint64_t cancelled_intervention = 0u;  // counterevidence path
  std::uint64_t cancelled_observation = 0u;   // counterevidence path
  bool context_retained = false;
  std::uint64_t context_mass_added = 0u;
  bool synapse_contacted_before = false;
};

__device__ inline ExperienceCreditReceipt assimilate_existing_experience_with_receipt(
    TissueView tissue, SparsePopulationView earlier, SparsePopulationView later_reference,
    SparsePopulationView context, std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present) {
  ExperienceCreditReceipt receipt{};
  if (earlier.count != 1u || later_reference.count != 1u || tissue.synapses == nullptr ||
      tissue.scalars == nullptr)
    return receipt;
  const SparseBindingSynapse* pre_synapse =
      find_synapse_read_only(tissue, earlier.cells[0], later_reference.cells[0]);
  if (pre_synapse == nullptr)
    return receipt;  // must already be resident -- refuses rather than invert a first claim
  const std::uint32_t synapse_index =
      static_cast<std::uint32_t>(pre_synapse - tissue.synapses);
  const PopulationCohortEvidence* pre_cohorts = synapse_cohorts(tissue, pre_synapse);
  if (pre_cohorts == nullptr)
    return receipt;
  std::uint32_t cohort_index = UINT32_MAX;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    if (pre_cohorts[index].claimed != 0u && same_cohort(pre_cohorts[index], earlier)) {
      cohort_index = index;
      break;
    }
  }
  if (cohort_index == UINT32_MAX)
    return receipt;  // would be a NEW cohort claim -- out of scope, refuse rather than mislabel

  const PopulationCohortEvidence before = pre_cohorts[cohort_index];
  const bool contacted_before = pre_synapse->contacted_since_drain != 0u;

  const bool accepted = assimilate_experience(tissue, earlier, later_reference, context,
                                              efferent_contact, efferent_polarity, outcome_present);
  if (!accepted)
    return receipt;  // nothing moved (e.g. an ineligible contact) -- nothing to invert

  const PopulationCohortEvidence& after =
      synapse_cohorts(tissue, tissue.synapses + synapse_index)[cohort_index];

  receipt.synapse_index = synapse_index;
  receipt.cohort_index = cohort_index;
  receipt.synapse_contacted_before = contacted_before;
  receipt.context_retained = after.qualifying_context_count != before.qualifying_context_count;
  if (receipt.context_retained)
    receipt.context_mass_added = after.context_mass - before.context_mass;

  // Classify by which field INCREASED -- the credited destination -- never merely "changed".
  // move_counter_mass (the counterevidence path) CANCELS FROM intervention_support first, so that
  // field can decrease during a counterevidence credit exactly as it increases during a concordant
  // intervention credit; checking "changed" alone misclassifies every counterevidence call whose
  // cancellation happens to land on intervention_support, which is the case this test's own LTD
  // reps hit every time (intervention_support has enough to cover all 8 discordant reps).
  if (after.counterevidence > before.counterevidence) {
    receipt.moved_counter = true;
    receipt.moved = after.counterevidence - before.counterevidence;
    receipt.cancelled_intervention = before.intervention_support - after.intervention_support;
    receipt.cancelled_observation = before.observational_support - after.observational_support;
  } else if (after.intervention_support > before.intervention_support) {
    receipt.moved_intervention = true;
    receipt.moved = after.intervention_support - before.intervention_support;
    receipt.cancelled_counter = before.counterevidence - after.counterevidence;
  } else if (after.observational_support > before.observational_support) {
    receipt.moved_observational = true;
    receipt.moved = after.observational_support - before.observational_support;
    receipt.cancelled_counter = before.counterevidence - after.counterevidence;
  }
  receipt.credited = true;
  return receipt;
}

// The exact reverse of ONE assimilate_existing_experience_with_receipt call. Applying every
// receipt from a sequence of forward calls, in EXACT REVERSE ORDER, returns the synapse/cohort/
// scalars state touched by those calls to its pre-sequence value byte-for-byte -- verified by
// bcc32_cuda_resident_plasticity_cycle_contract's own arm 9 against arm 1's 8 LTP and arm 2's 8
// LTD repetitions. This does not attempt to reverse apply_unreinforced_support_drain (a separate,
// tissue-wide operation) or apply_synapse_turnover_pressure/lesion_synapse_kernel (destructive
// without an escrow of what they overwrote) -- both remain named RED.
__device__ inline bool withdraw_experience(TissueView tissue,
                                           const ExperienceCreditReceipt& receipt) {
  if (!receipt.credited || tissue.synapses == nullptr || tissue.scalars == nullptr ||
      receipt.synapse_index >= tissue.synapse_capacity)
    return false;
  SparseBindingSynapse* synapse = tissue.synapses + receipt.synapse_index;
  if (synapse->claimed == 0u)
    return false;
  PopulationCohortEvidence* cohort = synapse_cohorts(tissue, synapse) + receipt.cohort_index;
  if (cohort->claimed == 0u)
    return false;

  if (receipt.moved_observational) {
    cohort->observational_support -= receipt.moved;
    cohort->counterevidence += receipt.cancelled_counter;
    tissue.scalars->free_mass += receipt.moved - receipt.cancelled_counter;
    --tissue.scalars->accepted_observations;
  } else if (receipt.moved_intervention) {
    cohort->intervention_support -= receipt.moved;
    cohort->counterevidence += receipt.cancelled_counter;
    tissue.scalars->free_mass += receipt.moved - receipt.cancelled_counter;
    --tissue.scalars->accepted_interventions;
  } else if (receipt.moved_counter) {
    cohort->counterevidence -= receipt.moved;
    cohort->intervention_support += receipt.cancelled_intervention;
    cohort->observational_support += receipt.cancelled_observation;
    tissue.scalars->free_mass +=
        receipt.moved - receipt.cancelled_intervention - receipt.cancelled_observation;
    --tissue.scalars->accepted_counterevidence;
  }

  if (receipt.context_retained) {
    --cohort->qualifying_context_count;
    const std::uint32_t slot = cohort->qualifying_context_count;
    for (std::uint32_t index = 0u; index < kMaximumPopulationCells; ++index)
      cohort->qualifying_context_cells[slot][index] = 0u;
    cohort->qualifying_context_counts[slot] = 0u;
    cohort->context_mass -= receipt.context_mass_added;
    tissue.scalars->free_mass += receipt.context_mass_added;
  }

  synapse->contacted_since_drain = receipt.synapse_contacted_before ? 1u : 0u;
  --tissue.scalars->revision;
  return true;
}

__device__ inline bool cohort_is_qualified(
    const PopulationCohortEvidence& cohort, SparsePopulationView active_context,
    CompletionPolicy policy) {
  const bool context_matches = active_context.count == 0u ||
                               cohort_has_context(cohort, active_context);
  return context_matches && qualified_for_completion(cohort, policy);
}

__device__ inline bool source_is_witnessed(
    TissueView tissue, std::uint32_t source, std::uint32_t target,
    SparsePopulationView active_context, CompletionPolicy policy) {
  const SparseBindingSynapse* synapse = find_synapse_read_only(tissue, source, target);
  if (synapse == nullptr || synapse->cohort_overflow_mass != 0u)
    return false;
  const PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
  if (cohorts == nullptr)
    return false;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index)
    if (cohorts[index].claimed != 0u &&
        cohort_is_qualified(cohorts[index], active_context, policy))
      return true;
  return false;
}

__device__ inline SelectedCohortEvidence select_coherent_forward_evidence(
    TissueView tissue, const SparseBindingSynapse& synapse,
    SparsePopulationView cue, SparsePopulationView active_context,
    CompletionPolicy policy) {
  SelectedCohortEvidence selected{};
  if (synapse.cohort_overflow_mass != 0u) {
    selected.ambiguous = 1u;
    return selected;
  }
  std::uint32_t stable_core[kMaximumPopulationCells]{};
  std::uint32_t stable_count = 0u;
  bool first = true;
  const PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, &synapse);
  if (cohorts == nullptr)
    return selected;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    const PopulationCohortEvidence& cohort = cohorts[index];
    if (cohort.claimed == 0u)
      continue;
    const std::uint64_t support =
        cohort.observational_support + cohort.intervention_support;
    if (!cohort_is_qualified(cohort, active_context, policy)) {
      if (selected.qualified == 0u && support > selected.support) {
        selected.support = support;
        selected.counterevidence = cohort.counterevidence;
      }
      continue;
    }
    if (first) {
      stable_count = cohort.cell_count;
      for (std::uint32_t cell = 0u; cell < stable_count; ++cell)
        stable_core[cell] = cohort.cells[cell];
      first = false;
    } else {
      std::uint32_t retained = 0u;
      for (std::uint32_t cell = 0u; cell < stable_count; ++cell)
        if (cohort_contains(cohort, stable_core[cell]))
          stable_core[retained++] = stable_core[cell];
      stable_count = retained;
    }
    const std::uint32_t overlap = cohort_overlap(cohort, cue);
    if (selected.qualified == 0u || overlap > selected.overlap ||
        (overlap == selected.overlap &&
         net_evidence(support, cohort.counterevidence) >
             net_evidence(selected.support, selected.counterevidence))) {
      selected.support = support;
      selected.counterevidence = cohort.counterevidence;
      selected.overlap = overlap;
      selected.qualified = 1u;
    }
  }
  if (first) {
    selected.qualified = 0u;
    return selected;
  }
  // A missing all-cohort intersection is uncertainty between learned source
  // populations, not absence of evidence.  Keep the resident overlap/evidence
  // winner graded above; stable-core guards still apply whenever a core exists.
  if (stable_count == 0u)
    return selected;
  bool core_is_subset = true;
  for (std::uint32_t index = 0u; index < stable_count; ++index)
    core_is_subset = core_is_subset && population_contains(cue, stable_core[index]);
  bool cue_is_subset = true;
  for (std::uint32_t index = 0u; index < cue.count; ++index) {
    bool present = false;
    for (std::uint32_t core = 0u; core < stable_count; ++core)
      present = present || cue.cells[index] == stable_core[core];
    cue_is_subset = cue_is_subset && present;
  }
  // A foreign cue must not gain authority merely by containing the small
  // stable core of another proposition.  A strict resident-population
  // majority may generalize across previously unseen surface cells; below
  // that consensus, every extra cell needs independent qualified matter for
  // this same target.  A genuine partial cue remains lawful through the
  // cue-is-subset path below, while a close or oversized distractor with only
  // coincidental overlap does not inherit the proposition.
  if (core_is_subset && !cue_is_subset) {
    const bool resident_majority = stable_count * 2u > cue.count;
    bool extras_witnessed = true;
    for (std::uint32_t index = 0u; index < cue.count; ++index) {
      bool in_core = false;
      for (std::uint32_t core = 0u; core < stable_count; ++core)
        in_core = in_core || cue.cells[index] == stable_core[core];
      if (!in_core)
        extras_witnessed = extras_witnessed && source_is_witnessed(
            tissue, cue.cells[index], synapse.target_cell, active_context,
            policy);
    }
    if (!resident_majority && !extras_witnessed)
      selected.qualified = 0u;
  }
  if (!core_is_subset && cue_is_subset) {
    for (std::uint32_t index = 0u; index < cue.count; ++index)
      cue_is_subset = cue_is_subset && source_is_witnessed(
          tissue, cue.cells[index], synapse.target_cell, active_context, policy);
  }
  if (!core_is_subset && !cue_is_subset)
    selected.qualified = 0u;
  return selected;
}

__device__ inline SelectedCohortEvidence select_cohort_evidence(
    TissueView tissue, const SparseBindingSynapse& synapse, SparsePopulationView cue,
    SparsePopulationView active_context, std::uint32_t reciprocal,
    CompletionPolicy policy) {
  SelectedCohortEvidence selected{};
  if (synapse.cohort_overflow_mass != 0u) {
    selected.ambiguous = 1u;
    return selected;
  }
  const PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, &synapse);
  if (cohorts == nullptr)
    return selected;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    const PopulationCohortEvidence& cohort = cohorts[index];
    if (cohort.claimed == 0u)
      continue;
    const std::uint32_t overlap =
        reciprocal != 0u ? 1u : cohort_overlap(cohort, cue);
    if (overlap == 0u)
      continue;
    const std::uint64_t support =
        cohort.observational_support + cohort.intervention_support;
    const bool context_matches = active_context.count == 0u ||
                                 cohort_has_context(cohort, active_context);
    const bool qualified = context_matches && qualified_for_completion(cohort, policy);
    if (reciprocal != 0u) {
      if (qualified &&
          (selected.qualified == 0u ||
           net_evidence(support, cohort.counterevidence) >
               net_evidence(selected.support, selected.counterevidence))) {
        selected.support = support;
        selected.counterevidence = cohort.counterevidence;
        selected.overlap = overlap;
        selected.qualified = 1u;
      } else if (selected.qualified == 0u && support > selected.support) {
        selected.support = support;
        selected.counterevidence = cohort.counterevidence;
      }
      continue;
    }
    if (overlap > selected.overlap) {
      selected.support = support;
      selected.counterevidence = cohort.counterevidence;
      selected.overlap = overlap;
      selected.qualified = qualified;
      selected.ambiguous = 0u;
    } else if (overlap == selected.overlap) {
      if (selected.qualified != static_cast<std::uint32_t>(qualified)) {
        selected.ambiguous = 1u;
      } else if (net_evidence(support, cohort.counterevidence) >
                 net_evidence(selected.support, selected.counterevidence)) {
        selected.support = support;
        selected.counterevidence = cohort.counterevidence;
      }
    }
  }
  return selected;
}

__device__ inline bool settle_completion(
    TissueView tissue, SparsePopulationView cue,
    SparsePopulationView active_context, std::uint32_t reciprocal,
    CompletionWorkspaceView workspace, CompletionResult* result,
    CompletionPolicy policy = CompletionPolicy::causal) {
  if (result == nullptr)
    return false;
  *result = CompletionResult{};
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr ||
      workspace.cell_scores == nullptr ||
      workspace.output_cells == nullptr || workspace.output_scores == nullptr ||
      workspace.cell_capacity < tissue.cell_capacity || workspace.output_capacity == 0u ||
      !valid_population(cue, tissue.cell_capacity) ||
      (active_context.count != 0u &&
       !valid_population(active_context, tissue.cell_capacity)))
    return false;
  for (std::uint32_t cell = 0u; cell < workspace.cell_capacity; ++cell)
    workspace.cell_scores[cell] = 0u;
  for (std::uint32_t index = 0u; index < tissue.synapse_capacity; ++index) {
    const SparseBindingSynapse& synapse = tissue.synapses[index];
    if (synapse.claimed == 0u)
      continue;
    const std::uint32_t cue_cell = reciprocal != 0u ? synapse.target_cell : synapse.source_cell;
    if (!population_contains(cue, cue_cell))
      continue;
    const SelectedCohortEvidence selected = reciprocal != 0u
        ? select_cohort_evidence(tissue, synapse, cue, active_context, reciprocal, policy)
        : select_coherent_forward_evidence(tissue, synapse, cue, active_context, policy);
    if (selected.overlap > result->cue_cells_matched)
      result->cue_cells_matched = selected.overlap;
    if (selected.qualified == 0u || selected.ambiguous != 0u) {
      result->uncertain_mass += selected.support;
      continue;
    }
    const std::uint32_t completed_cell =
        reciprocal != 0u ? synapse.source_cell : synapse.target_cell;
    const std::uint64_t previous = workspace.cell_scores[completed_cell];
    const std::uint64_t added_support = selected.support - selected.counterevidence;
    workspace.cell_scores[completed_cell] =
        added_support > UINT64_MAX - previous ? UINT64_MAX : previous + added_support;
    ++result->qualified_synapses;
  }
  const std::uint32_t output_count = workspace.output_capacity < tissue.cell_capacity
                                         ? workspace.output_capacity
                                         : tissue.cell_capacity;
  for (std::uint32_t rank = 0u; rank < output_count; ++rank) {
    std::uint32_t best_cell = 0u;
    std::uint64_t best_score = 0u;
    for (std::uint32_t cell = 0u; cell < tissue.cell_capacity; ++cell) {
      const std::uint64_t score = workspace.cell_scores[cell];
      if (score > best_score || (score == best_score && score != 0u && cell < best_cell)) {
        best_cell = cell;
        best_score = score;
      }
    }
    if (best_score == 0u)
      break;
    workspace.output_cells[result->output_count] = best_cell;
    workspace.output_scores[result->output_count] = best_score;
    workspace.cell_scores[best_cell] = 0u;
    if (best_score > result->strongest_score)
      result->strongest_score = best_score;
    ++result->output_count;
  }
  result->ready = result->output_count != 0u;
  result->tissue_revision = tissue.scalars->revision;
  return true;
}

__device__ inline bool settle_completion(
    TissueView tissue, SparsePopulationView cue, std::uint32_t reciprocal,
    CompletionWorkspaceView workspace, CompletionResult* result,
    CompletionPolicy policy = CompletionPolicy::causal) {
  return settle_completion(tissue, cue, SparsePopulationView{}, reciprocal,
                           workspace, result, policy);
}

// Compatibility entry for the context-conditioned commitment organ.  Its
// original callers remain causal by default; production discourse settlement
// opts into the weaker observation-backed policy explicitly.
__device__ inline void settle_completion_device(
    TissueView tissue, SparsePopulationView cue,
    SparsePopulationView active_context, std::uint32_t reciprocal,
    CompletionWorkspaceView workspace, CompletionResult* result,
    CompletionPolicy policy = CompletionPolicy::causal) {
  settle_completion(tissue, cue, active_context, reciprocal, workspace,
                    result, policy);
}

__global__ void settle_completion_kernel(TissueView tissue, SparsePopulationView cue,
                                         std::uint32_t reciprocal,
                                         CompletionWorkspaceView workspace,
                                         CompletionResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  settle_completion(tissue, cue, reciprocal, workspace, result,
                    CompletionPolicy::causal);
}

__global__ void settle_discourse_completion_kernel(
    TissueView tissue, SparsePopulationView cue, std::uint32_t reciprocal,
    CompletionWorkspaceView workspace, CompletionResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  settle_completion(tissue, cue, reciprocal, workspace, result,
                    CompletionPolicy::discourse);
}

__global__ void lesion_synapse_kernel(TissueView tissue, std::uint32_t synapse_index) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || tissue.scalars == nullptr ||
      tissue.synapses == nullptr || tissue.cohorts == nullptr ||
      synapse_index >= tissue.synapse_capacity)
    return;
  SparseBindingSynapse* synapse = tissue.synapses + synapse_index;
  if (synapse->claimed != 0u)
    return_synapse_mass(tissue, synapse);
  ++tissue.scalars->lesion_revision;
}

__global__ void lesion_fraction_kernel(TissueView tissue, std::uint32_t denominator,
                                       std::uint32_t numerator) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || denominator == 0u || numerator > denominator ||
      tissue.scalars == nullptr || tissue.synapses == nullptr || tissue.cohorts == nullptr)
    return;
  for (std::uint32_t index = 0u; index < tissue.synapse_capacity; ++index) {
    SparseBindingSynapse* synapse = tissue.synapses + index;
    if (synapse->claimed != 0u &&
        binding_hash(synapse->source_cell, synapse->target_cell) % denominator < numerator)
      return_synapse_mass(tissue, synapse);
  }
  ++tissue.scalars->lesion_revision;
}

__global__ void audit_mass_kernel(TissueView tissue, std::uint64_t* observed_mass) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || observed_mass == nullptr ||
      tissue.scalars == nullptr || tissue.synapses == nullptr || tissue.cohorts == nullptr)
    return;
  std::uint64_t total = tissue.scalars->free_mass;
  for (std::uint32_t index = 0u; index < tissue.synapse_capacity; ++index) {
    const SparseBindingSynapse& synapse = tissue.synapses[index];
    total += synapse.structure_mass + synapse.overflow_mass + synapse.cohort_overflow_mass;
    for (std::uint32_t cohort_index = 0u;
         cohort_index < kMaximumCohortsPerSynapse; ++cohort_index) {
      const PopulationCohortEvidence& cohort =
          tissue.cohorts[index * kMaximumCohortsPerSynapse + cohort_index];
      total += cohort.observational_support + cohort.intervention_support +
               cohort.counterevidence + cohort.structure_mass + cohort.context_mass;
    }
  }
  *observed_mass = total;
}

}  // namespace bcc32_cuda_resident_proposition_tissue
