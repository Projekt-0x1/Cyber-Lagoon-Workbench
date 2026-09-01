// Device-resident surface realization kernels and their bounded transaction
// helpers. This file is included inside the adult stream namespace after the
// stream state and all resident surface dependencies are declared.

inline constexpr std::uint32_t kFrozenSurfaceClosure = 1u << 31u;
inline constexpr std::uint32_t kClosureAlreadyPresent = 1u << 8u;
inline constexpr std::uint32_t kClosureMustAppend = 1u << 9u;
inline constexpr std::uint32_t kRelationSurfaceInactive = 0u;
inline constexpr std::uint32_t kRelationSurfaceAnswerMapping = 1u;
inline constexpr std::uint32_t kRelationSurfaceExactEvent = 2u;
inline constexpr std::uint32_t kRelationSurfaceProjected = 3u;

#include "bcc32_cuda_adult_stream_surface_handoff.inl"

__global__ void realize_ordered_relation_surface_step_kernel(
    surface_organ::SurfaceUnitView units,
    surface_organ::OpaqueConstructionWitnessView witness,
    const std::uint32_t* selection,
    const std::uint32_t* anchors, const std::uint32_t* anchor_count,
    surface_organ::SurfaceRealizationWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *workspace.result = {};
  surface_organ::surface_clear_output(workspace);
  if (anchors == nullptr || anchor_count == nullptr) return;
  const std::uint32_t count = anchor_count[0];
  if (count == 0u || count > surface_organ::kSurfaceOrganMaxAnchors) return;
  const surface_organ::OpaqueContentPlanView plan{anchors, count};
  if (selection != nullptr &&
      selection[2] != kRelationSurfaceInactive) {
    const std::uint32_t construction = selection[0];
    const std::uint32_t total =
        witness.count == nullptr ? 0u : min(witness.count[0], witness.capacity);
    if (construction >= total || witness.tokens == nullptr ||
        witness.lengths == nullptr || witness.slot_counts == nullptr ||
        witness.closed_class_mask == nullptr ||
        witness.slot_counts[construction] != count)
      return;
    const std::uint32_t extent = witness.lengths[construction];
    if (extent == 0u ||
        extent > adult::construction::kConstructionMaxTokens ||
        extent > workspace.output_unit_capacity)
      return;
    std::uint32_t slot = 0u;
    for (std::uint32_t position = 0u; position < extent; ++position) {
      const std::uint32_t token =
          witness.tokens[construction *
                             adult::construction::kConstructionMaxTokens +
                         position];
      const bool is_slot = adult::construction::token_is_slot(token);
      const std::uint32_t unit = is_slot ? anchors[slot++] : token;
      if (unit >= units.unit_count)
        return;
      workspace.output_units[position] = unit;
      workspace.output_anchor_mask[position] = is_slot ? 1u : 0u;
    }
    if (slot != count) return;
    std::uint32_t byte_count = 0u;
    if (!surface_organ::surface_emit_bytes(units, workspace, extent,
                                           &byte_count))
      return;
    // selection[3] is frozen by pre-contact projection. Live closure or class
    // learning from the current query must not authorize its answer surface.
    const std::uint32_t frozen_closure = selection[3];
    if ((frozen_closure & kFrozenSurfaceClosure) == 0u || byte_count == 0u)
      return;
    if ((frozen_closure & kClosureMustAppend) != 0u) {
      if (byte_count >= workspace.output_byte_capacity) return;
      workspace.output_bytes[byte_count++] =
          static_cast<std::uint8_t>(frozen_closure & 0xffu);
    }
    workspace.result->ready = 1u;
    workspace.result->grammar_supported = 1u;
    workspace.result->closure_supported = 1u;
    workspace.result->anchors_preserved = count;
    workspace.result->output_unit_count = extent;
    workspace.result->output_byte_count = byte_count;
    workspace.result->construction_count = total;
    workspace.result->construction_supported = 1u;
    workspace.result->construction_shape_matched = 1u;
    workspace.result->construction_mapping_matched = 1u;
    workspace.result->path_quality_q20 =
        witness.supports == nullptr ? 0u : witness.supports[construction];
  } else {
    surface_organ::realize_ordered_construction_witness(units, plan, witness,
                                                         workspace);
  }
}

// Select the sole learned construction written by the same retained event as
// the grounded relation plan. Similar shapes from other contacts are not
// substitutes, and an equal event witness remains an abstention.
[[nodiscard]] __device__ inline bool same_answer_surface(
    std::uint32_t left, std::uint32_t right,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_slot_counts) {
  if (left == right) return true;
  if (construction_lengths[left] != construction_lengths[right] ||
      construction_slot_counts[left] != construction_slot_counts[right])
    return false;
  for (std::uint32_t position = 0u;
       position < construction_lengths[left]; ++position) {
    const std::uint32_t left_token =
        construction_tokens[
            left * adult::construction::kConstructionMaxTokens + position];
    const std::uint32_t right_token =
        construction_tokens[
            right * adult::construction::kConstructionMaxTokens + position];
    if (adult::construction::token_is_slot(left_token) &&
        adult::construction::token_is_slot(right_token))
      continue;
    if (left_token != right_token) return false;
  }
  return true;
}

__global__ void select_relation_surface_witness_kernel(
    const std::uint64_t* selected_evidence_revision,
    const adult::construction::WitnessedRelationEvent* events,
    const std::uint32_t* event_cursor,
    const std::uint32_t* event_constructions,
    const std::uint32_t* event_surface_units,
    const std::uint32_t* event_surface_counts,
    const std::uint32_t* cue_exact,
    const std::uint64_t* qonset_evidence_revision,
    const std::uint32_t* question_gap_field_support,
    const std::uint32_t* question_answer_construction,
    const std::uint32_t* question_answer_construction_support,
    const std::uint32_t* question_answer_slot_mapping,
    const std::uint32_t* role_canon, std::uint32_t unit_count,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_slot_counts,
    const std::uint32_t* construction_closed_class_mask,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const std::uint32_t* construction_count, std::uint32_t capacity,
    std::uint32_t* selection, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || selection == nullptr) return;
  selection[0] = adult::construction::kNoConstruction;
  selection[1] = 0xffffffffu;
  selection[2] = 0u;
  selection[3] = 0u;
  if (selected_evidence_revision != nullptr &&
      selected_evidence_revision[0] != 0u && receipt != nullptr)
    ++receipt->relation_surface_events;
  if (selected_evidence_revision == nullptr ||
      events == nullptr || event_cursor == nullptr ||
      event_constructions == nullptr || event_surface_units == nullptr ||
      event_surface_counts == nullptr || construction_tokens == nullptr ||
      construction_lengths == nullptr ||
      construction_slot_counts == nullptr ||
      construction_closed_class_mask == nullptr ||
      construction_slot_units == nullptr ||
      construction_slot_masses == nullptr || construction_count == nullptr ||
      selected_evidence_revision[0] == 0u)
    return;
  const std::uint32_t event_extent = min(
      event_cursor[0], adult::construction::kWitnessedRelationEventCap);
  const std::uint32_t event_first =
      event_cursor[0] > adult::construction::kWitnessedRelationEventCap
          ? event_cursor[0] &
                (adult::construction::kWitnessedRelationEventCap - 1u)
          : 0u;
  std::uint32_t target_event = 0xffffffffu;
  for (std::uint32_t offset = 0u; offset < event_extent; ++offset) {
    const std::uint32_t index =
        (event_first + offset) &
        (adult::construction::kWitnessedRelationEventCap - 1u);
    if (events[index].live == 0u ||
        events[index].evidence_revision != selected_evidence_revision[0])
      continue;
    if (target_event != 0xffffffffu) return;
    target_event = index;
  }
  if (target_event == 0xffffffffu) {
    if (receipt != nullptr) ++receipt->relation_surface_missing_events;
    return;
  }
  const std::uint32_t target_fields[adult::construction::kRelationFieldCount] = {
      events[target_event].triple.subject,
      events[target_event].triple.connective,
      events[target_event].triple.connective2,
      events[target_event].triple.value};
  std::uint32_t target_arity = 0u;
  for (std::uint32_t field = 0u;
       field < adult::construction::kRelationFieldCount; ++field)
    target_arity +=
        target_fields[field] != adult::construction::kNoTripleUnit;

  std::uint32_t answer_construction = adult::construction::kNoConstruction;
  std::uint32_t answer_slot_mapping = 0u;
  const std::uint32_t total = min(construction_count[0], capacity);
  if (cue_exact != nullptr && qonset_evidence_revision != nullptr &&
      question_gap_field_support != nullptr &&
      question_answer_construction != nullptr &&
      question_answer_construction_support != nullptr &&
      question_answer_slot_mapping != nullptr) {
    std::uint32_t opener = adult::construction::kNoTripleUnit;
    for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
      if (cue_exact[unit] == 0u) continue;
      const std::uint32_t canon =
          role_canon != nullptr && role_canon[unit] < unit_count
              ? role_canon[unit]
              : unit;
      if (qonset_evidence_revision[canon] == 0u) continue;
      if (opener != adult::construction::kNoTripleUnit && opener != canon) {
        opener = adult::construction::kNoTripleUnit;
        break;
      }
      opener = canon;
    }
    if (opener != adult::construction::kNoTripleUnit) {
      std::uint32_t requested = adult::construction::kRelationFieldCount;
      std::uint32_t best_support = 0u;
      bool tied = false;
      for (std::uint32_t field = 0u;
           field < adult::construction::kRelationFieldCount; ++field) {
        const std::uint32_t support =
            question_gap_field_support[
                opener * adult::construction::kRelationFieldCount + field];
        if (support > best_support) {
          requested = field;
          best_support = support;
          tied = false;
        } else if (support != 0u && support == best_support) {
          tied = true;
        }
      }
      if (!tied && requested < adult::construction::kRelationFieldCount) {
        const std::size_t mapping =
            adult::construction::question_answer_surface_index(
                opener, requested, target_arity);
        if (question_answer_construction_support[mapping] >=
            adult::construction::kConstructionMinRoleEvidence) {
          answer_construction = question_answer_construction[mapping];
          answer_slot_mapping = question_answer_slot_mapping[mapping];
        }
      }
    }
  }
  // Declarative answer order is reusable across interrogative forms. If this
  // opener has not yet acquired its own answer witness, use the sole recurrent
  // surface learned at the target relation arity. Conflicting learned surfaces
  // abstain; no lexical opener or authored role chooses among them.
  if (answer_construction == adult::construction::kNoConstruction &&
      target_arity != 0u &&
      target_arity <= adult::construction::kConstructionMaxSlots &&
      question_answer_construction != nullptr &&
      question_answer_construction_support != nullptr &&
      question_answer_slot_mapping != nullptr) {
    bool surface_ambiguous = false;
    for (std::uint32_t candidate_opener = 0u;
         candidate_opener < unit_count && !surface_ambiguous;
         ++candidate_opener) {
      for (std::uint32_t field = 0u;
           field < adult::construction::kRelationFieldCount; ++field) {
        const std::size_t mapping =
            adult::construction::question_answer_surface_index(
                candidate_opener, field, target_arity);
        if (question_answer_construction_support[mapping] <
            adult::construction::kConstructionMinRoleEvidence)
          continue;
        const std::uint32_t candidate =
            question_answer_construction[mapping];
        const std::uint32_t candidate_mapping =
            question_answer_slot_mapping[mapping];
        if (candidate >= total || candidate_mapping == 0u) continue;
        if (answer_construction == adult::construction::kNoConstruction) {
          answer_construction = candidate;
          answer_slot_mapping = candidate_mapping;
          continue;
        }
        surface_ambiguous |=
            answer_slot_mapping != candidate_mapping ||
            !same_answer_surface(answer_construction, candidate,
                                 construction_tokens, construction_lengths,
                                 construction_slot_counts);
      }
    }
    if (surface_ambiguous) {
      answer_construction = adult::construction::kNoConstruction;
      answer_slot_mapping = 0u;
    }
  }
  selection[3] = answer_slot_mapping;
  std::uint32_t selected = adult::construction::kNoConstruction;
  std::uint32_t selected_event = 0xffffffffu;
  bool matched_event = false;
  for (std::uint32_t offset = 0u; offset < event_extent; ++offset) {
    const std::uint32_t index =
        (event_first + offset) &
        (adult::construction::kWitnessedRelationEventCap - 1u);
    if (events[index].live == 0u ||
        events[index].evidence_revision != selected_evidence_revision[0])
      continue;
    if (selected_event != 0xffffffffu) return;
    selected_event = index;
    matched_event = true;
    std::uint32_t construction =
        answer_construction != adult::construction::kNoConstruction &&
                answer_construction !=
                    adult::construction::kAmbiguousConstruction
            ? answer_construction
            : event_constructions[index];
    if (construction == answer_construction && construction < total) {
      bool relation_covers_surface = true;
      const std::uint32_t surface_count = event_surface_counts[index];
      for (std::uint32_t position = 0u;
           position < surface_count && relation_covers_surface; ++position) {
        const std::uint32_t unit =
            event_surface_units[
                index * adult::construction::kConstructionMaxTokens +
                position];
        if (unit >= unit_count ||
            construction_closed_class_mask[unit] != 0u)
          continue;
        bool represented = false;
        for (std::uint32_t field = 0u;
             field < adult::construction::kRelationFieldCount; ++field)
          represented |= target_fields[field] == unit;
        relation_covers_surface = represented;
      }
      if (target_arity != construction_slot_counts[construction] ||
          !relation_covers_surface) {
        construction = event_constructions[index];
        answer_construction = adult::construction::kNoConstruction;
      }
    }
    if (construction >= total) {
      const std::uint32_t surface_count = event_surface_counts[index];
      std::uint32_t recovered = adult::construction::kNoConstruction;
      if (surface_count != 0u &&
          surface_count <= adult::construction::kConstructionMaxTokens) {
        for (std::uint32_t candidate = 0u; candidate < total; ++candidate) {
          if (construction_lengths[candidate] != surface_count) continue;
          bool exact = true;
          std::uint32_t slot = 0u;
          for (std::uint32_t position = 0u;
               position < surface_count && exact; ++position) {
            const std::uint32_t token =
                construction_tokens[
                    candidate *
                        adult::construction::kConstructionMaxTokens +
                    position];
            const std::uint32_t unit =
                event_surface_units[
                    index * adult::construction::kConstructionMaxTokens +
                    position];
            if (!adult::construction::token_is_slot(token)) {
              exact = token == unit;
              continue;
            }
            if (slot >= construction_slot_counts[candidate]) {
              exact = false;
              continue;
            }
            bool member = false;
            for (std::uint32_t population = 0u;
                 population <
                 adult::construction::kConstructionSlotPopulationCap;
                 ++population) {
              const std::size_t member_index =
                  adult::construction::construction_slot_member_index(
                      candidate, slot, population);
              member |= construction_slot_units[member_index] == unit &&
                        construction_slot_masses[member_index] != 0u;
            }
            exact = member;
            ++slot;
          }
          exact &= slot == construction_slot_counts[candidate];
          if (!exact) continue;
          if (recovered != adult::construction::kNoConstruction) return;
          recovered = candidate;
        }
      }
      construction = recovered;
    }
    if (construction >= total ||
        construction_lengths[construction] == 0u) {
      if (receipt != nullptr)
        ++receipt->relation_surface_missing_constructions;
      return;
    }
    if (selected != adult::construction::kNoConstruction &&
        selected != construction)
      return;
    selected = construction;
  }
  if (!matched_event) {
    if (receipt != nullptr) ++receipt->relation_surface_missing_events;
    return;
  }
  if (selected == adult::construction::kNoConstruction) {
    if (receipt != nullptr)
      ++receipt->relation_surface_missing_constructions;
    return;
  }
  selection[0] = selected;
  selection[1] = selected_event;
  const bool uses_answer_mapping =
      answer_construction != adult::construction::kNoConstruction &&
      answer_construction != adult::construction::kAmbiguousConstruction;
  selection[2] = uses_answer_mapping ? kRelationSurfaceAnswerMapping
                                     : kRelationSurfaceExactEvent;
  if (!uses_answer_mapping) selection[3] = 0u;
  if (receipt != nullptr) ++receipt->relation_surface_witnesses;
}

// Recover the construction's fillers from the exact retained relation event.
// Slot populations provide the only mapping authority. No role name, byte
// literal, or authored sentence order enters this bridge.
__global__ void stage_relation_surface_anchors_kernel(
    const std::uint64_t* selected_evidence_revision,
    const adult::construction::WitnessedRelationEvent* events,
    const std::uint32_t* event_cursor, const std::uint32_t* selection,
    const std::uint32_t* event_surface_units,
    const std::uint32_t* event_surface_counts,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_slot_counts,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const std::uint32_t* construction_slot_overflow,
    const adult::roles::MutableStructuralRole* roles,
    std::uint32_t unit_count,
    std::uint32_t* anchors, std::uint32_t anchor_capacity,
    std::uint32_t* anchor_count, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || anchor_count == nullptr) return;
  anchor_count[0] = 0u;
  if (selected_evidence_revision == nullptr || events == nullptr ||
      event_cursor == nullptr || selection == nullptr ||
      event_surface_units == nullptr || event_surface_counts == nullptr ||
      construction_tokens == nullptr || construction_lengths == nullptr ||
      construction_slot_counts == nullptr ||
      construction_slot_units == nullptr ||
      construction_slot_masses == nullptr || roles == nullptr ||
      anchors == nullptr ||
      selected_evidence_revision[0] == 0u ||
      selection[0] == adult::construction::kNoConstruction)
    return;
  const std::uint32_t extent =
      min(event_cursor[0], adult::construction::kWitnessedRelationEventCap);
  const std::uint32_t first =
      event_cursor[0] > adult::construction::kWitnessedRelationEventCap
          ? event_cursor[0] &
                (adult::construction::kWitnessedRelationEventCap - 1u)
          : 0u;
  const adult::construction::WitnessedRelationEvent* source = nullptr;
  std::uint32_t source_event = adult::construction::kNoTripleUnit;
  for (std::uint32_t offset = 0u; offset < extent; ++offset) {
    const std::uint32_t index =
        (first + offset) &
        (adult::construction::kWitnessedRelationEventCap - 1u);
    if (events[index].live != 0u &&
        events[index].evidence_revision == selected_evidence_revision[0]) {
      if (source != nullptr) return;
      source = events + index;
      source_event = index;
    }
  }
  if (source == nullptr) return;
  const std::uint32_t construction = selection[0];
  const std::uint32_t slots = construction_slot_counts[construction];
  if (slots == 0u || slots > anchor_capacity ||
      slots > adult::construction::kConstructionMaxSlots)
    return;
  if (selection[2] == kRelationSurfaceAnswerMapping) {
    const std::uint32_t fields[adult::construction::kRelationFieldCount] = {
        source->triple.subject, source->triple.connective,
        source->triple.connective2, source->triple.value};
    for (std::uint32_t slot = 0u; slot < slots; ++slot) {
      const std::uint32_t encoded = (selection[3] >> (slot * 3u)) & 7u;
      if (encoded == 0u ||
          encoded > adult::construction::kRelationFieldCount)
        return;
      const std::uint32_t unit = fields[encoded - 1u];
      if (unit == adult::construction::kNoTripleUnit || unit >= unit_count)
        return;
      anchors[slot] = unit;
    }
    anchor_count[0] = slots;
    if (receipt != nullptr) {
      ++receipt->relation_surface_anchor_frames;
      receipt->surface_trajectory_slots = slots;
      receipt->surface_trajectory_grounded = slots;
    }
    return;
  }
  const std::uint32_t source_count = event_surface_counts[source_event];
  if (source_count == 0u ||
      source_count > adult::construction::kConstructionMaxTokens)
    return;
  // Overflow is aggregate population state, not evidence against this retained
  // event. Exact-event staging still requires every filler to be present in
  // its learned slot below before closure can freeze the witness.
  (void)construction_slot_overflow;
  bool source_used[adult::construction::kConstructionMaxTokens] = {};
  for (std::uint32_t slot = 0u; slot < slots; ++slot) {
    std::uint32_t expected_role = adult::construction::kNoTripleUnit;
    std::uint32_t observed_slots = 0u;
    for (std::uint32_t token_index = 0u;
         token_index < construction_lengths[construction]; ++token_index) {
      const std::uint32_t token =
          construction_tokens[construction *
                                  adult::construction::kConstructionMaxTokens +
                              token_index];
      if (!adult::construction::token_is_slot(token)) continue;
      if (observed_slots++ == slot) {
        expected_role = adult::construction::token_role(token);
        break;
      }
    }
    if (expected_role == adult::construction::kNoTripleUnit) return;
    std::uint32_t selected_source = source_count;
    for (std::uint32_t source_index = 0u; source_index < source_count;
         ++source_index) {
      const std::uint32_t unit =
          event_surface_units[source_event *
                                  adult::construction::kConstructionMaxTokens +
                              source_index];
      if (source_used[source_index] ||
          unit == adult::construction::kNoTripleUnit)
        continue;
      bool member = false;
      for (std::uint32_t population_index = 0u;
           population_index <
           adult::construction::kConstructionSlotPopulationCap;
           ++population_index) {
        const std::size_t member_index =
            adult::construction::construction_slot_member_index(
                construction, slot, population_index);
        member |= construction_slot_units[member_index] == unit &&
                  construction_slot_masses[member_index] != 0u;
      }
      if (member) {
        selected_source = source_index;
        break;
      }
    }
    if (selected_source == source_count) return;
    source_used[selected_source] = true;
    anchors[slot] =
        event_surface_units[source_event *
                                adult::construction::kConstructionMaxTokens +
                            selected_source];
  }
  anchor_count[0] = slots;
  if (receipt != nullptr) {
    ++receipt->relation_surface_anchor_frames;
    receipt->surface_trajectory_slots = slots;
    receipt->surface_trajectory_grounded = slots;
  }
}

[[nodiscard]] __device__ inline std::uint64_t
learned_population_context_overlap(
    const population_surface::UnitPopulationView& populations,
    std::uint32_t left_unit, std::uint32_t right_unit) {
  if (populations.cells == nullptr ||
      populations.population_context_mass == nullptr ||
      left_unit >= populations.unit_count || right_unit >= populations.unit_count)
    return 0u;
  const std::uint32_t left_count =
      populations.population_count == nullptr
          ? populations.population_width
          : populations.population_count[left_unit];
  const std::uint32_t right_count =
      populations.population_count == nullptr
          ? populations.population_width
          : populations.population_count[right_unit];
  if (left_count == 0u || right_count == 0u ||
      left_count > populations.population_width ||
      right_count > populations.population_width)
    return 0u;
  const std::uint32_t* contextual_cells =
      populations.context_cells == nullptr ? populations.cells
                                           : populations.context_cells;
  const std::uint32_t* left =
      contextual_cells +
      static_cast<std::size_t>(left_unit) * populations.population_width;
  const std::uint32_t* right =
      contextual_cells +
      static_cast<std::size_t>(right_unit) * populations.population_width;
  const std::uint16_t* left_mass =
      populations.population_context_mass +
      static_cast<std::size_t>(left_unit) * populations.population_width;
  const std::uint16_t* right_mass =
      populations.population_context_mass +
      static_cast<std::size_t>(right_unit) * populations.population_width;
  std::uint64_t support = 0u;
  for (std::uint32_t left_slot = 0u; left_slot < left_count; ++left_slot) {
    if (left_mass[left_slot] == 0u) continue;
    for (std::uint32_t right_slot = 0u; right_slot < right_count; ++right_slot) {
      if (right_mass[right_slot] == 0u ||
          left[left_slot] != right[right_slot])
        continue;
      support += min(static_cast<std::uint32_t>(left_mass[left_slot]),
                     static_cast<std::uint32_t>(right_mass[right_slot]));
    }
  }
  return support;
}

[[nodiscard]] __device__ inline std::uint64_t
relation_field_slot_support(
    std::uint32_t construction, std::uint32_t slot, std::uint32_t field_unit,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const population_surface::UnitPopulationView& populations) {
  std::uint64_t support = 0u;
  for (std::uint32_t member = 0u;
       member < adult::construction::kConstructionSlotPopulationCap; ++member) {
    const std::size_t index =
        adult::construction::construction_slot_member_index(construction, slot,
                                                            member);
    const std::uint32_t member_unit = construction_slot_units[index];
    const std::uint32_t member_mass = construction_slot_masses[index];
    if (member_unit >= populations.unit_count || member_mass == 0u) continue;
    const std::uint64_t contextual =
        learned_population_context_overlap(populations, field_unit, member_unit);
    const std::uint64_t exact =
        field_unit == member_unit
            ? static_cast<std::uint64_t>(member_mass)
            : 0u;
    const std::uint64_t member_support =
        exact > contextual ? exact : contextual;
    if (member_support > support) support = member_support;
  }
  return support;
}

__device__ inline std::uint32_t projected_surface_learned_closure(
    std::uint32_t construction, const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths, const std::uint32_t* anchors,
    std::uint32_t anchor_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    std::uint32_t unit_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses) {
  if (construction_tokens == nullptr || construction_lengths == nullptr ||
      anchors == nullptr || unit_lengths == nullptr || unit_content == nullptr ||
      unit_words == 0u || boundary_mask == nullptr || closure_bytes == nullptr ||
      closure_count == 0u)
    return 0u;
  bool closure_supported = false;
  std::uint32_t supported_closure = 0u;
  std::uint32_t slot = 0u;
  const std::uint32_t extent = construction_lengths[construction];
  for (std::uint32_t position = 0u; position < extent; ++position) {
    const std::uint32_t token =
        construction_tokens[
            construction * adult::construction::kConstructionMaxTokens +
            position];
    const bool is_slot = adult::construction::token_is_slot(token);
    if (is_slot && slot >= anchor_count) return 0u;
    const std::uint32_t unit = is_slot ? anchors[slot++] : token;
    if (unit >= unit_count) return 0u;
    for (std::uint32_t byte = 0u; byte < unit_lengths[unit]; ++byte) {
      const std::uint32_t value = adult::construction::construction_unit_byte(
          unit_content, unit_words, unit, byte);
      bool learned_closure = false;
      for (std::uint32_t closure = 0u; closure < closure_count; ++closure)
        learned_closure |= closure_bytes[closure] <= 0xffu &&
                           value == closure_bytes[closure];
      if (learned_closure) {
        closure_supported = true;
        supported_closure = value;
      } else if (closure_supported &&
               (value > 0xffu || boundary_mask[value] == 0u))
        closure_supported = false;
    }
  }
  if (slot != anchor_count) return 0u;
  if (closure_supported)
    return kClosureAlreadyPresent | supported_closure;

  // Some acquired surfaces carried punctuation inside an exemplar slot. Slot
  // replacement must not copy that exemplar's content, but its recurrently
  // observed terminal byte remains learned construction glue.
  if (construction_slot_units == nullptr ||
      construction_slot_masses == nullptr)
    return 0u;
  std::uint32_t learned = 0u;
  bool found = false;
  for (std::uint32_t slot_index = 0u; slot_index < anchor_count; ++slot_index) {
    for (std::uint32_t member = 0u;
         member < adult::construction::kConstructionSlotPopulationCap;
         ++member) {
      const std::size_t index =
          adult::construction::construction_slot_member_index(
              construction, slot_index, member);
      const std::uint32_t unit = construction_slot_units[index];
      if (unit >= unit_count || construction_slot_masses[index] == 0u) continue;
      for (std::uint32_t byte = 0u; byte < unit_lengths[unit]; ++byte) {
        const std::uint32_t value =
            adult::construction::construction_unit_byte(
                unit_content, unit_words, unit, byte);
        bool is_closure = false;
        for (std::uint32_t closure = 0u; closure < closure_count; ++closure)
          is_closure |= closure_bytes[closure] <= 0xffu &&
                        value == closure_bytes[closure];
        if (!is_closure) continue;
        bool suffix_is_boundary = true;
        for (std::uint32_t suffix = byte + 1u;
             suffix < unit_lengths[unit]; ++suffix) {
          const std::uint32_t suffix_value =
              adult::construction::construction_unit_byte(
                  unit_content, unit_words, unit, suffix);
          suffix_is_boundary &=
              suffix_value <= 0xffu && boundary_mask[suffix_value] != 0u;
        }
        if (!suffix_is_boundary) continue;
        if (found && learned != value) return 0u;
        found = true;
        learned = value;
      }
    }
  }
  return found ? (kClosureMustAppend | learned) : 0u;
}

__global__ void freeze_selected_relation_surface_closure_kernel(
    const std::uint64_t* selected_evidence_revision,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* anchors, std::uint32_t* anchor_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit_count,
    const std::uint32_t* boundary_mask, const std::uint32_t* closure_bytes,
    std::uint32_t closure_count, const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses, std::uint32_t* selection,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || selection == nullptr ||
      anchor_count == nullptr || selection[2] == 0u ||
      selection[0] == adult::construction::kNoConstruction ||
      anchor_count[0] == 0u)
    return;
  const std::uint32_t closure = projected_surface_learned_closure(
      selection[0], construction_tokens, construction_lengths, anchors,
      anchor_count[0], unit_lengths, unit_content, unit_words, unit_count,
      boundary_mask, closure_bytes, closure_count, construction_slot_units,
      construction_slot_masses);
  if (closure == 0u) {
    selection[0] = adult::construction::kNoConstruction;
    selection[1] = 0xffffffffu;
    selection[2] = 0u;
    selection[3] = 0u;
    anchor_count[0] = 0u;
    return;
  }
  selection[3] = kFrozenSurfaceClosure | closure;
  if (receipt != nullptr) {
    if (selected_evidence_revision != nullptr &&
        selected_evidence_revision[0] != 0u)
      ++receipt->relation_surface_events;
    ++receipt->relation_surface_witnesses;
    ++receipt->relation_surface_anchor_frames;
    receipt->surface_trajectory_slots = anchor_count[0];
    receipt->surface_trajectory_grounded = anchor_count[0];
  }
}

// Project a selected declarative relation into a separately acquired surface
// construction. The fact supplies only opaque field populations; learned
// construction slots supply order. A complete, unique joint projection is
// required. Event source units and learned question-answer tables are absent
// from this interface and therefore cannot become response authority.
__global__ void project_relation_obligations_to_construction_kernel(
    const std::uint64_t* selected_evidence_revision,
    const adult::construction::WitnessedRelationEvent* events,
    const std::uint32_t* event_cursor,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_slot_counts,
    const std::uint32_t* construction_supports,
    const std::uint64_t* construction_origin_revision,
    const std::uint32_t* construction_closed_class_mask,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const std::uint32_t* construction_slot_overflow,
    const std::uint32_t* construction_count, std::uint32_t capacity,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    population_surface::UnitPopulationView populations,
    std::uint32_t* selection, std::uint32_t* anchors,
    std::uint32_t anchor_capacity, std::uint32_t* anchor_count,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || selection == nullptr ||
      anchor_count == nullptr)
    return;
  if (selection[2] != 0u &&
      (selection[3] & kFrozenSurfaceClosure) != 0u &&
      anchor_count[0] != 0u)
    return;
  selection[0] = adult::construction::kNoConstruction;
  selection[1] = 0xffffffffu;
  selection[2] = 0u;
  selection[3] = 0u;
  anchor_count[0] = 0u;
  if (selected_evidence_revision != nullptr &&
      selected_evidence_revision[0] != 0u && receipt != nullptr)
    ++receipt->relation_surface_events;
  if (selected_evidence_revision == nullptr || events == nullptr ||
      event_cursor == nullptr || construction_tokens == nullptr ||
      construction_lengths == nullptr ||
      construction_slot_counts == nullptr ||
      construction_supports == nullptr ||
      construction_origin_revision == nullptr ||
      construction_closed_class_mask == nullptr ||
      construction_slot_units == nullptr ||
      construction_slot_masses == nullptr ||
      construction_slot_overflow == nullptr ||
      construction_count == nullptr || unit_lengths == nullptr ||
      unit_content == nullptr || boundary_mask == nullptr ||
      closure_bytes == nullptr || closure_count == 0u || anchors == nullptr ||
      populations.cells == nullptr || populations.population_context_mass == nullptr ||
      selected_evidence_revision[0] == 0u)
    return;

  const std::uint32_t event_extent =
      min(event_cursor[0], adult::construction::kWitnessedRelationEventCap);
  const std::uint32_t event_first =
      event_cursor[0] > adult::construction::kWitnessedRelationEventCap
          ? event_cursor[0] &
                (adult::construction::kWitnessedRelationEventCap - 1u)
          : 0u;
  std::uint32_t target_event = 0xffffffffu;
  for (std::uint32_t offset = 0u; offset < event_extent; ++offset) {
    const std::uint32_t index =
        (event_first + offset) &
        (adult::construction::kWitnessedRelationEventCap - 1u);
    if (events[index].live == 0u ||
        events[index].evidence_revision != selected_evidence_revision[0])
      continue;
    if (target_event != 0xffffffffu) return;
    target_event = index;
  }
  if (target_event == 0xffffffffu) {
    if (receipt != nullptr) ++receipt->relation_surface_missing_events;
    return;
  }

  const std::uint32_t raw_fields[adult::construction::kRelationFieldCount] = {
      events[target_event].triple.subject,
      events[target_event].triple.connective,
      events[target_event].triple.connective2,
      events[target_event].triple.value};
  std::uint32_t fields[adult::construction::kRelationFieldCount]{};
  std::uint32_t field_count = 0u;
  for (std::uint32_t field = 0u;
       field < adult::construction::kRelationFieldCount; ++field) {
    const std::uint32_t unit = raw_fields[field];
    if (unit >= populations.unit_count ||
        unit == adult::construction::kNoTripleUnit ||
        construction_closed_class_mask[unit] != 0u)
      continue;
    fields[field_count++] = unit;
  }
  if (field_count == 0u || field_count > anchor_capacity ||
      field_count > adult::construction::kConstructionMaxSlots)
    return;

  const std::uint32_t total = min(construction_count[0], capacity);
  std::uint32_t best_construction = adult::construction::kNoConstruction;
  std::uint32_t best_anchors[adult::construction::kConstructionMaxSlots]{};
  std::uint64_t best_score = 0u;
  std::uint32_t best_surface_support = 0u;
  std::uint32_t best_closure_code = 0u;
  bool tied = false;
  for (std::uint32_t construction = 0u; construction < total; ++construction) {
    if (construction_lengths[construction] == 0u ||
        construction_slot_counts[construction] != field_count ||
        (construction_origin_revision[construction] >> 32u) ==
            (selected_evidence_revision[0] >> 32u))
      continue;
    selection[3] = max(selection[3], 1u);
    bool literal_valid = true;
    for (std::uint32_t position = 0u;
         position < construction_lengths[construction]; ++position) {
      const std::uint32_t token =
          construction_tokens[
              construction * adult::construction::kConstructionMaxTokens +
              position];
      literal_valid &= adult::construction::token_is_slot(token) ||
                       (token < populations.unit_count &&
                        construction_closed_class_mask[token] != 0u);
    }
    for (std::uint32_t slot = 0u; slot < field_count; ++slot)
      literal_valid &=
          construction_slot_overflow[
              adult::construction::construction_slot_index(construction, slot)] ==
          0u;
    if (!literal_valid) continue;
    selection[3] = max(selection[3], 2u);

    std::uint32_t combinations = 1u;
    for (std::uint32_t slot = 0u; slot < field_count; ++slot)
      combinations *= field_count;
    for (std::uint32_t code = 0u; code < combinations; ++code) {
      std::uint32_t cursor = code;
      std::uint32_t used = 0u;
      std::uint32_t projected[adult::construction::kConstructionMaxSlots]{};
      // Construction frequency admits no answer and broad slot populations
      // receive no additive advantage. Each obligation contributes only its
      // strongest learned member edge.
      std::uint64_t score = 0u;
      bool complete = true;
      for (std::uint32_t slot = 0u; slot < field_count; ++slot) {
        const std::uint32_t field = cursor % field_count;
        cursor /= field_count;
        if ((used & (1u << field)) != 0u) {
          complete = false;
          break;
        }
        used |= 1u << field;
        const std::uint64_t slot_support = relation_field_slot_support(
            construction, slot, fields[field], construction_slot_units,
            construction_slot_masses, populations);
        if (slot_support == 0u) {
          complete = false;
          break;
        }
        score += slot_support;
        projected[slot] = fields[field];
      }
      if (!complete) continue;
      selection[3] = max(selection[3], 3u);
      const std::uint32_t closure_code =
          projected_surface_learned_closure(
              construction, construction_tokens, construction_lengths,
              projected, field_count, unit_lengths, unit_content, unit_words,
              populations.unit_count, boundary_mask, closure_bytes,
              closure_count, construction_slot_units,
              construction_slot_masses);
      if (closure_code == 0u)
        continue;
      selection[3] = max(selection[3], 4u);
      bool same_joint = construction == best_construction;
      for (std::uint32_t slot = 0u; slot < field_count && same_joint; ++slot)
        same_joint &= projected[slot] == best_anchors[slot];
      const std::uint32_t surface_support =
          construction_supports[construction];
      if (best_construction == adult::construction::kNoConstruction ||
          score > best_score ||
          (score == best_score && surface_support > best_surface_support)) {
        best_construction = construction;
        best_score = score;
        best_surface_support = surface_support;
        best_closure_code = closure_code;
        tied = false;
        for (std::uint32_t slot = 0u; slot < field_count; ++slot)
          best_anchors[slot] = projected[slot];
      } else if (score == best_score &&
                 surface_support == best_surface_support && !same_joint) {
        tied = true;
      }
    }
  }

  if (best_construction == adult::construction::kNoConstruction || tied) {
    if (receipt != nullptr)
      ++receipt->relation_surface_missing_constructions;
    if (tied && receipt != nullptr)
      receipt->surface_trajectory_ambiguous = 1u;
    return;
  }
  selection[0] = best_construction;
  selection[1] = target_event;
  selection[2] = kRelationSurfaceProjected;
  selection[3] = kFrozenSurfaceClosure | best_closure_code;
  for (std::uint32_t slot = 0u; slot < field_count; ++slot)
    anchors[slot] = best_anchors[slot];
  anchor_count[0] = field_count;
  if (receipt != nullptr) {
    ++receipt->relation_surface_witnesses;
    ++receipt->relation_surface_anchor_frames;
    receipt->surface_trajectory_slots = field_count;
    receipt->surface_trajectory_grounded = field_count;
  }
}

[[nodiscard]] __host__ __device__ inline std::uint64_t recent_topic_key(
    std::uint64_t evidence_revision, std::uint32_t cue_order,
    std::uint32_t unit, std::uint32_t cue_hits = 1u) {
  const std::uint32_t contact_revision =
      static_cast<std::uint32_t>(evidence_revision >> 32u);
  const std::uint32_t order =
      cue_order < 0xffffu ? cue_order : static_cast<std::uint32_t>(0xffffu);
  const std::uint32_t bounded_hits = cue_hits < 15u ? cue_hits : 15u;
  return (static_cast<std::uint64_t>(bounded_hits) << 60u) |
         (static_cast<std::uint64_t>(contact_revision & 0x0fffffffu) << 32u) |
         (static_cast<std::uint64_t>(0xffffu - order) << 16u) |
         static_cast<std::uint64_t>(0xffffu - unit);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t
recent_topic_hits(std::uint64_t key) {
  return static_cast<std::uint32_t>(key >> 60u);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t
recent_topic_revision(std::uint64_t key) {
  return static_cast<std::uint32_t>((key >> 32u) & 0x0fffffffu);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t
recent_topic_unit(std::uint64_t key) {
  return key == 0u
             ? adult::construction::kNoTripleUnit
             : 0xffffu - static_cast<std::uint32_t>(key & 0xffffu);
}

// Accumulate the latest retained witnessed relation for each exact cue
// endpoint. The tournament runs before the current contact is assimilated, so
// recency is resident discourse evidence rather than query self-confirmation.
__global__ void accumulate_recent_exact_content_relation_support_kernel(
    const adult::construction::WitnessedRelationEvent* events,
    const std::uint32_t* event_cursor, const std::uint32_t* cue_exact,
    const std::uint32_t* cue_orders, const std::uint32_t* closed_class_mask,
    const std::uint32_t* event_surface_units,
    const std::uint32_t* event_surface_counts,
    const std::uint32_t* qonset_count,
    std::uint32_t unit_count, std::uint64_t* support) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (events == nullptr || event_cursor == nullptr || cue_exact == nullptr ||
      cue_orders == nullptr || closed_class_mask == nullptr ||
      support == nullptr)
    return;
  (void)event_surface_units;
  (void)event_surface_counts;
  (void)qonset_count;
  const std::uint32_t extent =
      min(event_cursor[0], adult::construction::kWitnessedRelationEventCap);
  if (offset >= extent) return;
  const std::uint32_t first =
      event_cursor[0] > adult::construction::kWitnessedRelationEventCap
          ? event_cursor[0] &
                (adult::construction::kWitnessedRelationEventCap - 1u)
          : 0u;
  const std::uint32_t event_index =
      (first + offset) &
      (adult::construction::kWitnessedRelationEventCap - 1u);
  const auto& event = events[event_index];
  if (event.live == 0u || event.evidence_revision == 0u) return;
  // Request modality is resolved before the event enters this store. Reusing
  // a mutable onset statistic here can later relabel an observed fact as a
  // request and erase it from retrieval.
  const auto triple = event.triple;
  const std::uint32_t fields[5] = {
      triple.subject, triple.connective, triple.connective2, triple.value,
      event.terminal};
  std::uint32_t cue_hits = 0u;
  for (std::uint32_t field = 0u; field < 5u; ++field) {
    const std::uint32_t unit = fields[field];
    if (unit >= unit_count || cue_exact[unit] == 0u ||
        closed_class_mask[unit] != 0u)
      continue;
    bool duplicate = false;
    for (std::uint32_t prior = 0u; prior < field; ++prior)
      duplicate |= fields[prior] == unit;
    if (!duplicate) ++cue_hits;
  }
  if (cue_hits == 0u) return;
  const std::uint32_t endpoints[2] = {triple.subject, triple.value};
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const std::uint32_t unit = endpoints[index];
    if (unit >= unit_count || cue_exact[unit] == 0u ||
        cue_orders[unit] == adult::construction::kNoTripleUnit ||
        closed_class_mask[unit] != 0u)
      continue;
    const std::uint64_t key =
        recent_topic_key(event.evidence_revision, cue_orders[unit], unit,
                         cue_hits);
    atomicMax(reinterpret_cast<unsigned long long*>(support + unit),
              static_cast<unsigned long long>(key));
  }
}

// Prefer the newest learned relation-bearing contact, then the earliest exact
// endpoint in the current cue. The packed key remains a metric-only device
// tournament; no byte, word, role, or host label receives authored authority.
__global__ void derive_supported_exact_content_key_kernel(
    const std::uint64_t* support, const std::uint32_t* cue_exact,
    const std::uint32_t* cue_orders, const std::uint32_t* closed_class_mask,
    std::uint32_t unit_count, std::uint64_t* topic_key) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count || support == nullptr || cue_exact == nullptr ||
      cue_orders == nullptr || closed_class_mask == nullptr ||
      topic_key == nullptr ||
      cue_exact[unit] == 0u || closed_class_mask[unit] != 0u ||
      cue_orders[unit] == adult::construction::kNoTripleUnit || support[unit] == 0u)
    return;
  const std::uint64_t key = support[unit];
  atomicMax(reinterpret_cast<unsigned long long*>(topic_key),
            static_cast<unsigned long long>(key));
}

__global__ void publish_specific_topic_unit_kernel(
    const std::uint64_t* topic_key, std::uint32_t* topic_unit) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || topic_key == nullptr ||
      topic_unit == nullptr)
    return;
  topic_unit[0] = recent_topic_unit(topic_key[0]);
}

__global__ void initialize_stream_surface_span_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    SurfaceSpanTransaction* transaction, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transaction == nullptr) return;
  // A grounded relation commitment was emitted first. It already owns the
  // motor buffer; a later construction pass may inspect its own witness but
  // must not erase that device-selected answer.
  *transaction = {};
  if (generated_count != nullptr && generated_count[0] != 0u) return;
  if (plan == nullptr || !discourse_plan::valid(*plan) ||
      plan->status != discourse_plan::PlanStatus::committed)
    return;
  const std::uint32_t span =
      discourse_plan::dependency_linked_step_span(*plan, plan->active_step);
  if (span == 0u) return;
  transaction->first_step = plan->active_step;
  transaction->expected_steps = span;
}

__global__ void initialize_relation_surface_span_kernel(
    const std::uint32_t* evidence_count, const std::uint32_t* selections,
    SurfaceSpanTransaction* transaction) {
  if (transaction == nullptr) return;
  if (threadIdx.x == 0u) *transaction = {};
  __syncthreads();
  if (evidence_count == nullptr || evidence_count[0] == 0u ||
      evidence_count[0] > adult::construction::kRelationSurfaceEvidenceCap ||
      selections == nullptr)
    return;
  std::uint32_t local_expected_steps = 0u;
  for (std::uint32_t step = threadIdx.x; step < evidence_count[0];
       step += blockDim.x) {
    const std::uint32_t* selection = selections + step * 4u;
    local_expected_steps += selection[2] != 0u &&
                                    (selection[3] & kFrozenSurfaceClosure) != 0u
                                ? 1u
                                : 0u;
  }
  if (local_expected_steps != 0u)
    atomicAdd(&transaction->expected_steps, local_expected_steps);
}

// Append only the next required member of the dependency-linked span.  Any
// missing, ambiguous, or capacity-failing frame invalidates the complete
// transaction before motor emission; no partial prefix becomes observable.
__global__ void append_stream_surface_step_kernel(
    std::uint32_t step, const surface_organ::SurfaceOrganResult* realization,
    surface_organ::SurfaceUnitView units, const std::uint32_t* surface_units,
    std::uint32_t surface_unit_capacity,
    SurfaceSpanTransaction* transaction, std::uint8_t* candidate,
    std::uint32_t candidate_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transaction == nullptr ||
      transaction->expected_steps == 0u || transaction->failed != 0u)
    return;
  if (transaction->realized_steps >= transaction->expected_steps) return;
  const std::uint32_t expected_step =
      transaction->first_step + transaction->realized_steps;
  if (step != expected_step) return;
  if (realization == nullptr || surface_units == nullptr || candidate == nullptr ||
      realization->ready == 0u || realization->grammar_supported == 0u ||
      realization->output_byte_count == 0u ||
      realization->output_unit_count == 0u ||
      realization->output_unit_count > surface_unit_capacity ||
      realization->output_byte_count > candidate_capacity - transaction->byte_count) {
    transaction->failed = 1u;
    return;
  }
  // Serialize the exact device-selected unit sequence.  The renderer has
  // already validated the witness, anchor mapping, and byte extent; avoiding
  // its transient byte staging buffer keeps the same learned realization while
  // making the transaction's motor bytes the selected units themselves.
  std::uint32_t written = 0u;
  for (std::uint32_t index = 0u; index < realization->output_unit_count; ++index) {
    const std::uint32_t unit = surface_units[index];
    if (unit >= units.unit_count || units.lengths[unit] > units.unit_words * 4u ||
        written > realization->output_byte_count ||
        units.lengths[unit] > realization->output_byte_count - written) {
      transaction->failed = 1u;
      return;
    }
    for (std::uint32_t offset = 0u; offset < units.lengths[unit]; ++offset) {
      const std::uint32_t word =
          units.packed_bytes[static_cast<std::size_t>(unit) * units.unit_words + offset / 4u];
      candidate[transaction->byte_count + written++] =
          static_cast<std::uint8_t>((word >> ((offset % 4u) * 8u)) & 0xffu);
    }
  }
  if (written != realization->output_byte_count) {
    transaction->failed = 1u;
    return;
  }
  transaction->byte_count += written;
  transaction->anchor_count += realization->anchors_preserved;
  transaction->construction_count += realization->construction_count;
  transaction->construction_supported += realization->construction_supported;
  transaction->construction_shape_matched += realization->construction_shape_matched;
  transaction->construction_mapping_matched += realization->construction_mapping_matched;
  transaction->construction_tied += realization->construction_tied;
  ++transaction->realized_steps;
}

// Event frames are independently exact. An event without a lawful surface
// witness abstains locally; it does not erase another event that was grounded
// and realized. Capacity failure remains transaction-fatal so no truncated
// frame can reach the motor buffer.
__global__ void append_relation_surface_step_kernel(
    const std::uint64_t* evidence_revision,
    const surface_organ::SurfaceOrganResult* realization,
    surface_organ::SurfaceUnitView units, const std::uint32_t* surface_units,
    std::uint32_t surface_unit_capacity, const std::uint8_t* surface_bytes,
    const std::uint32_t* boundary_mask, const std::uint32_t* boundary_bytes,
    SurfaceSpanTransaction* transaction, std::uint8_t* candidate,
    std::uint32_t candidate_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transaction == nullptr ||
      transaction->failed != 0u || realization == nullptr ||
      surface_units == nullptr || surface_bytes == nullptr ||
      candidate == nullptr || boundary_mask == nullptr)
    return;
  if (realization->ready == 0u || realization->grammar_supported == 0u ||
      realization->output_byte_count == 0u ||
      realization->output_unit_count == 0u ||
      realization->output_unit_count > surface_unit_capacity)
    return;
  if (evidence_revision == nullptr || evidence_revision[0] == 0u) return;
  for (std::uint32_t prior = 0u;
       prior < transaction->emitted_evidence_count; ++prior)
    if (transaction->emitted_evidence[prior] == evidence_revision[0]) return;
  if (transaction->emitted_evidence_count >=
      adult::construction::kRelationSurfaceEvidenceCap)
    return;

  const std::uint32_t frame_count = realization->output_unit_count;
  for (std::uint32_t index = 0u; index < frame_count; ++index) {
    const std::uint32_t unit = surface_units[index];
    if (unit >= units.unit_count ||
        units.lengths[unit] > units.unit_words * 4u) {
      ++transaction->rejected_steps;
      transaction->last_rejection = 1u;
      return;
    }
  }

  // A selected event can be represented by another selected event's longer
  // exact frame. Suppress only that contiguous sequence, never a repeated
  // content identity elsewhere in the answer.
  bool subsumed = false;
  if (frame_count <= transaction->emitted_unit_count) {
    for (std::uint32_t begin = 0u;
         begin + frame_count <= transaction->emitted_unit_count;
         ++begin) {
      bool same = true;
      for (std::uint32_t index = 0u; index < frame_count; ++index)
        same &= transaction->emitted_units[begin + index] ==
                surface_units[index];
      subsumed |= same;
    }
  }
  if (subsumed) {
    transaction->emitted_evidence[transaction->emitted_evidence_count++] =
        evidence_revision[0];
    transaction->anchor_count += realization->anchors_preserved;
    transaction->construction_count += realization->construction_count;
    transaction->construction_supported +=
        realization->construction_supported;
    transaction->construction_shape_matched +=
        realization->construction_shape_matched;
    transaction->construction_mapping_matched +=
        realization->construction_mapping_matched;
    transaction->construction_tied += realization->construction_tied;
    ++transaction->realized_steps;
    return;
  }

  // Merge only a physical sequence overlap: the longest suffix already
  // emitted that equals the new frame's prefix. This preserves legitimate
  // repeated entities and predicates while removing duplicate overlap such
  // as "... opened" + "opened the ...".
  std::uint32_t overlap = 0u;
  const std::uint32_t overlap_limit =
      min(transaction->emitted_unit_count, frame_count);
  for (std::uint32_t width = 1u; width <= overlap_limit; ++width) {
    bool same = true;
    for (std::uint32_t index = 0u; index < width; ++index)
      same &= transaction->emitted_units[
                  transaction->emitted_unit_count - width + index] ==
              surface_units[index];
    if (same) overlap = width;
  }
  const std::uint32_t appended_units = frame_count - overlap;
  if (transaction->emitted_unit_count + appended_units >
      surface_organ::kSurfaceOrganMaxOutputUnits) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 2u;
    return;
  }
  const bool tail_is_closure = transaction->closure_supported != 0u;
  // A non-overlapping frame can continue when its evidence is later in the
  // same witnessed contact. The acquisition revision is the physical sequence
  // authority; limiting this exception to the first continuation silently
  // truncated every learned trajectory after two independently realized
  // frames. Cross-contact residual fragments still require an observed
  // closure.
  bool same_contact_forward = false;
  if (transaction->emitted_evidence_count != 0u) {
    const std::uint64_t previous =
        transaction->emitted_evidence[transaction->emitted_evidence_count - 1u];
    same_contact_forward =
        (previous >> 32u) == (evidence_revision[0] >> 32u) &&
        previous < evidence_revision[0];
  }
  if (transaction->emitted_unit_count != 0u && overlap == 0u &&
      !tail_is_closure && !same_contact_forward) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 3u;
    return;
  }
  std::uint32_t skipped_bytes = 0u;
  for (std::uint32_t index = 0u; index < overlap; ++index)
    skipped_bytes += units.lengths[surface_units[index]];
  if (skipped_bytes > realization->output_byte_count) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 4u;
    return;
  }
  const std::uint32_t required_bytes =
      realization->output_byte_count - skipped_bytes;
  if (transaction->byte_count > candidate_capacity ||
      required_bytes > candidate_capacity - transaction->byte_count) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 4u;
    return;
  }
  bool needs_boundary = false;
  if (required_bytes != 0u && overlap == 0u &&
      transaction->byte_count != 0u &&
      boundary_bytes != nullptr && boundary_bytes[0] <= 0xffu &&
      boundary_mask != nullptr && boundary_mask[boundary_bytes[0]] != 0u) {
    needs_boundary = tail_is_closure;
    const std::uint32_t first_unit = surface_units[0];
    if (units.lengths[first_unit] != 0u) {
      const std::uint32_t first_word =
          units.packed_bytes[
              static_cast<std::size_t>(first_unit) * units.unit_words];
      const std::uint8_t first_byte =
          static_cast<std::uint8_t>(first_word & 0xffu);
      if (boundary_mask[first_byte] != 0u) needs_boundary = false;
    }
  }
  if (needs_boundary &&
      transaction->byte_count + required_bytes >= candidate_capacity) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 5u;
    return;
  }
  std::uint32_t written = 0u;
  if (needs_boundary)
    candidate[transaction->byte_count + written++] =
        static_cast<std::uint8_t>(boundary_bytes[0]);
  for (std::uint32_t byte = skipped_bytes;
       byte < realization->output_byte_count; ++byte)
    candidate[transaction->byte_count + written++] = surface_bytes[byte];
  for (std::uint32_t index = overlap; index < frame_count; ++index) {
    const std::uint32_t unit = surface_units[index];
    transaction->emitted_units[transaction->emitted_unit_count++] = unit;
  }
  if (written == 0u) {
    ++transaction->rejected_steps;
    transaction->last_rejection = 6u;
    return;
  }
  transaction->emitted_evidence[transaction->emitted_evidence_count++] =
      evidence_revision[0];
  transaction->byte_count += written;
  transaction->closure_supported = realization->closure_supported;
  transaction->anchor_count += realization->anchors_preserved;
  transaction->construction_count += realization->construction_count;
  transaction->construction_supported += realization->construction_supported;
  transaction->construction_shape_matched +=
      realization->construction_shape_matched;
  transaction->construction_mapping_matched +=
      realization->construction_mapping_matched;
  transaction->construction_tied += realization->construction_tied;
  ++transaction->realized_steps;
}

__global__ void finalize_relation_surface_span_kernel(
    SurfaceSpanTransaction* transaction,
    surface_organ::SurfaceOrganResult* surface_result,
    QueryAnswerReceipt* receipt, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transaction == nullptr ||
      surface_result == nullptr || generated_count == nullptr)
    return;
  if (generated_count[0] != 0u) return;
  *surface_result = {};
  generated_count[0] = 0u;
  const bool complete = transaction->expected_steps != 0u &&
      transaction->failed == 0u && transaction->rejected_steps == 0u &&
      transaction->realized_steps == transaction->expected_steps &&
      transaction->byte_count != 0u &&
      transaction->closure_supported != 0u;
  if (receipt != nullptr) {
    receipt->construction_count = transaction->construction_count;
    receipt->construction_supported = transaction->construction_supported;
    receipt->construction_shape_matched =
        transaction->construction_shape_matched;
    receipt->construction_mapping_matched =
        transaction->construction_mapping_matched;
    receipt->construction_tied = transaction->construction_tied;
    receipt->construction_ready = complete ? 1u : 0u;
    receipt->tx_expected_steps = transaction->expected_steps;
    receipt->tx_failed = transaction->failed;
    receipt->tx_realized_steps = transaction->realized_steps;
    receipt->tx_byte_count = transaction->byte_count;
    receipt->construction_grammar_supported = complete ? 1u : 0u;
    receipt->construction_output_bytes =
        complete ? transaction->byte_count : 0u;
    receipt->surface_trajectory_slots = transaction->anchor_count;
    receipt->surface_trajectory_grounded = transaction->anchor_count;
    receipt->surface_trajectory_ambiguous =
        transaction->failed != 0u ? 1u : 0u;
    receipt->serialized_units =
        complete ? transaction->emitted_unit_count : 0u;
  }
  if (!complete) return;
  surface_result->ready = 1u;
  surface_result->grammar_supported = 1u;
  surface_result->closure_supported = transaction->closure_supported;
  surface_result->anchors_preserved = transaction->anchor_count;
  surface_result->output_unit_count = transaction->emitted_unit_count;
  surface_result->output_byte_count = transaction->byte_count;
  surface_result->construction_count = transaction->construction_count;
  surface_result->construction_supported =
      transaction->construction_supported;
  surface_result->construction_shape_matched =
      transaction->construction_shape_matched;
  surface_result->construction_mapping_matched =
      transaction->construction_mapping_matched;
  surface_result->construction_tied = transaction->construction_tied;
  generated_count[0] = transaction->byte_count;
}

__global__ void finalize_stream_surface_span_kernel(
    SurfaceSpanTransaction* transaction,
    surface_organ::SurfaceOrganResult* surface_result,
    QueryAnswerReceipt* receipt, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transaction == nullptr ||
      surface_result == nullptr || generated_count == nullptr)
    return;
  if (generated_count[0] != 0u) return;
  *surface_result = {};
  generated_count[0] = 0u;
  const bool complete = transaction->expected_steps != 0u &&
      transaction->failed == 0u &&
      transaction->realized_steps == transaction->expected_steps &&
      transaction->byte_count != 0u;
  if (receipt != nullptr) {
    receipt->construction_count = transaction->construction_count;
    receipt->construction_supported = transaction->construction_supported;
    receipt->construction_shape_matched = transaction->construction_shape_matched;
    receipt->construction_mapping_matched = transaction->construction_mapping_matched;
    receipt->construction_tied = transaction->construction_tied;
    receipt->construction_ready = complete ? 1u : 0u;
    receipt->construction_grammar_supported = complete ? 1u : 0u;
    receipt->construction_output_bytes = complete ? transaction->byte_count : 0u;
    receipt->surface_trajectory_slots = transaction->anchor_count;
    receipt->surface_trajectory_grounded = transaction->anchor_count;
    receipt->surface_trajectory_ambiguous = transaction->failed != 0u ? 1u : 0u;
    receipt->serialized_units = complete ? transaction->anchor_count : 0u;
  }
  if (!complete) return;
  surface_result->ready = 1u;
  surface_result->grammar_supported = 1u;
  surface_result->closure_supported = 1u;
  surface_result->anchors_preserved = transaction->anchor_count;
  surface_result->output_unit_count = transaction->anchor_count;
  surface_result->output_byte_count = transaction->byte_count;
  surface_result->construction_count = transaction->construction_count;
  surface_result->construction_supported = transaction->construction_supported;
  surface_result->construction_shape_matched = transaction->construction_shape_matched;
  surface_result->construction_mapping_matched = transaction->construction_mapping_matched;
  surface_result->construction_tied = transaction->construction_tied;
  generated_count[0] = transaction->byte_count;
}

__global__ void adopt_stream_construction_surface_kernel(
    const surface_organ::SurfaceOrganResult* realization,
    const std::uint8_t* surface_bytes, std::uint32_t surface_capacity,
    QueryAnswerReceipt* receipt, std::uint8_t* candidate,
    std::uint32_t candidate_capacity, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || realization == nullptr ||
      receipt == nullptr)
    return;
  receipt->construction_count = realization->construction_count;
  receipt->construction_supported = realization->construction_supported;
  receipt->construction_shape_matched = realization->construction_shape_matched;
  receipt->construction_mapping_matched = realization->construction_mapping_matched;
  receipt->construction_tied = realization->construction_tied;
  receipt->construction_ready = realization->ready;
  receipt->construction_grammar_supported = realization->grammar_supported;
  receipt->construction_output_bytes = realization->output_byte_count;
  receipt->surface_closure_supported = realization->closure_supported;
  receipt->surface_anchors_preserved = realization->anchors_preserved;
  receipt->surface_capacity_exceeded = realization->capacity_exceeded;
  receipt->surface_output_units = realization->output_unit_count;
  receipt->surface_connectors = realization->connector_count;
  receipt->surface_selected_permutation = realization->selected_permutation;
  if (surface_bytes == nullptr || candidate == nullptr || generated_count == nullptr ||
      generated_count[0] != 0u ||
      receipt->attempted == 0u || realization->ready == 0u ||
      realization->grammar_supported == 0u ||
      realization->output_byte_count == 0u ||
      realization->output_byte_count > surface_capacity ||
      realization->output_byte_count > candidate_capacity)
    return;
  for (std::uint32_t index = 0u; index < realization->output_byte_count; ++index)
    candidate[index] = surface_bytes[index];
  generated_count[0] = realization->output_byte_count;
  receipt->serialized_units = realization->output_unit_count;
}

__global__ void realize_selected_action_surface_kernel(
    const PendingActionTrajectory* trace, QueryAnswerReceipt* receipt,
    std::uint8_t* candidate, std::uint32_t candidate_capacity,
    std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || trace == nullptr ||
      receipt == nullptr || candidate == nullptr || generated_count == nullptr ||
      receipt->attempted == 0u || trace->action_byte_count == 0u ||
      trace->action_byte_count > candidate_capacity || generated_count[0] != 0u)
    return;
  for (std::uint32_t index = 0u; index < trace->action_byte_count; ++index)
    candidate[index] = trace->action_surface[index];
  *generated_count = trace->action_byte_count;
  receipt->serialized_units = 1u;
}
