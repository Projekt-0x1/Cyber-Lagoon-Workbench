// Device-owned handoff from a committed discourse plan to the resident
// construction surface.  This file is included inside the stream namespace.

__global__ void stage_stream_surface_plan_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    const std::uint32_t* selection,
    const bcc32_cuda_resident_proposition_tissue::OrderedRoleBindingEvidence* bindings,
    const std::uint32_t* binding_constructions,
    const std::uint32_t* construction_slot_counts,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const std::uint32_t* construction_slot_overflow,
    const std::uint64_t* construction_evidence_revision,
    std::uint32_t binding_capacity, std::uint32_t construction_capacity,
    std::uint32_t* anchors,
    std::uint32_t anchor_capacity,
    std::uint32_t* anchor_count, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || anchor_count == nullptr) return;
  anchor_count[0] = 0u;
  if (receipt != nullptr) {
    receipt->surface_trajectory_slots = 0u;
    receipt->surface_trajectory_grounded = 0u;
    receipt->surface_trajectory_ambiguous = 0u;
  }
  if (plan == nullptr || selection == nullptr || anchors == nullptr ||
      !discourse_plan::valid(*plan) ||
      plan->status != discourse_plan::PlanStatus::committed ||
      selection[1] >= plan->step_count)
    return;
  const auto& step = plan->steps[selection[1]];

  // Opaque population plans already carry resident-selected learned surface
  // units in the committed Plan.  Copy only that bounded evidence extent;
  // construction selection remains in realize_construction_witness(), which
  // applies the resident support/uniqueness/tie law.  In particular, do not
  // require an ordered binding or let the current contact populate this span.
  if (step.reference_kind == discourse_plan::PlanReferenceKind::opaque_population) {
    if (step.population_count == 0u || step.anchor_count == 0u ||
        step.anchor_begin > plan->anchor_reference_count ||
        step.anchor_count > plan->anchor_reference_count - step.anchor_begin ||
        step.anchor_count > anchor_capacity ||
        step.anchor_count > surface_organ::kSurfaceOrganMaxAnchors)
      return;
    for (std::uint32_t index = 0u; index < step.anchor_count; ++index)
      anchors[index] = plan->anchor_references[step.anchor_begin + index];
    anchor_count[0] = step.anchor_count;
    if (receipt != nullptr) {
      receipt->surface_trajectory_slots = step.anchor_count;
      receipt->surface_trajectory_grounded = step.anchor_count;
    }
    return;
  }

  if (selection[0] >= construction_capacity || bindings == nullptr ||
      binding_constructions == nullptr || construction_slot_counts == nullptr ||
      construction_slot_units == nullptr || construction_slot_masses == nullptr ||
      construction_evidence_revision == nullptr ||
      step.reference_kind != discourse_plan::PlanReferenceKind::ordered_binding ||
      step.population_count != 1u ||
      step.population_begin >= plan->population_reference_count)
    return;
  const std::uint32_t focal = plan->population_references[step.population_begin];
  if (focal >= binding_capacity ||
      binding_constructions[focal] != selection[0])
    return;
  const auto& focal_binding = bindings[focal];
  const std::uint32_t construction = selection[0];
  const std::uint32_t slots = construction_slot_counts[construction];
  // A surface frame is assembled only from role identities carried by this
  // exact committed Plan step. Learned slot mass resolves competing retained
  // identities; exact equal mass remains ambiguous. A dependency-linked plan
  // is realized as an ordered span of such frames; it is never flattened into
  // one terminal construction or allowed to borrow a prior step's filler.
  // `step.evidence_revision` was captured before this contact enters
  // plasticity. Do not reread a mutable binding revision here: the incoming
  // contact may strengthen that binding after its response has already been
  // selected, but may never replace the evidence that authorizes this plan.
  const std::uint64_t evidence_revision = step.evidence_revision;
  if (focal_binding.claimed == 0u || evidence_revision == 0u ||
      (construction_evidence_revision[construction] >> 32u) !=
          (evidence_revision >> 32u) ||
      slots == 0u || slots > anchor_capacity ||
      slots > adult::construction::kConstructionMaxSlots)
    return;
  if (receipt != nullptr) receipt->surface_trajectory_slots = slots;
  for (std::uint32_t slot = 0u; slot < slots; ++slot) {
    if (construction_slot_overflow != nullptr &&
        construction_slot_overflow[adult::construction::construction_slot_index(
            construction, slot)] != 0u)
      return;
    std::uint32_t resolved = 0xffffffffu;
    bool ambiguous = false;
    std::uint32_t best_mass = 0u;
    for (std::uint32_t role = 0u;
         role < bcc32_cuda_resident_proposition_tissue::kOrderedBindingRoleCount;
         ++role) {
      if (focal_binding.role_unit_counts[role] == 0u ||
          focal_binding.role_unit_counts[role] >
              bcc32_cuda_resident_proposition_tissue::kMaximumOrderedRoleUnits)
        continue;
      for (std::uint32_t member = 0u;
           member < adult::construction::kConstructionSlotPopulationCap;
           ++member) {
        const std::size_t member_index =
            adult::construction::construction_slot_member_index(
                construction, slot, member);
        const std::uint32_t mass = construction_slot_masses[member_index];
        if (mass == 0u) continue;
        const std::uint32_t unit = construction_slot_units[member_index];
        bool retained = false;
        for (std::uint32_t identity = 0u;
             identity < focal_binding.role_unit_counts[role]; ++identity)
          retained |= focal_binding.role_units[role][identity] == unit;
        if (!retained) continue;
        if (resolved == 0xffffffffu || mass > best_mass) {
          resolved = unit;
          best_mass = mass;
          ambiguous = false;
        } else if (mass == best_mass && resolved != unit) {
          ambiguous = true;
        }
      }
    }
    if (ambiguous || resolved == 0xffffffffu) {
      if (receipt != nullptr && ambiguous)
        receipt->surface_trajectory_ambiguous = 1u;
      return;
    }
    anchors[slot] = resolved;
    if (receipt != nullptr) ++receipt->surface_trajectory_grounded;
  }
  anchor_count[0] = slots;
}

__global__ void select_stream_surface_witness_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    const std::uint32_t* binding_constructions,
    std::uint32_t binding_capacity, std::uint32_t requested_step,
    std::uint32_t* selection) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || selection == nullptr) return;
  selection[0] = adult::construction::kNoConstruction;
  selection[1] = 0xffffffffu;
  if (plan == nullptr || !discourse_plan::valid(*plan) ||
      plan->status != discourse_plan::PlanStatus::committed ||
      requested_step >= plan->step_count)
    return;
  const auto& candidate = plan->steps[requested_step];
  if (candidate.reference_kind ==
      discourse_plan::PlanReferenceKind::opaque_population) {
    if (candidate.population_count == 0u || candidate.anchor_count == 0u ||
        candidate.anchor_begin > plan->anchor_reference_count ||
        candidate.anchor_count >
            plan->anchor_reference_count - candidate.anchor_begin)
      return;
    // Keep construction selection opaque. The resident realization selector
    // below will scan learned constructions and abstain on unsupported or
    // tied mappings; no binding/construction table is consulted here.
    selection[1] = requested_step;
    return;
  }
  if (binding_constructions == nullptr ||
      candidate.reference_kind !=
          discourse_plan::PlanReferenceKind::ordered_binding ||
      candidate.population_count != 1u ||
      candidate.population_begin >= plan->population_reference_count)
    return;
  const std::uint32_t binding =
      plan->population_references[candidate.population_begin];
  if (binding >= binding_capacity) return;
  const std::uint32_t linked = binding_constructions[binding];
  if (linked == adult::construction::kNoConstruction) return;
  selection[0] = linked;
  selection[1] = requested_step;
}

// Surface realization remains device-owned even though the renderer is
// launched from the stream. The count is read from resident staging matter;
// the host never branches on a step's identities or chooses a witness.
__global__ void realize_stream_surface_step_kernel(
    surface_organ::SurfaceUnitView units,
    surface_organ::OpaqueConstructionWitnessView witness,
    const std::uint32_t* anchors, const std::uint32_t* anchor_count,
    surface_organ::SurfaceRealizationWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *workspace.result = {};
  surface_organ::surface_clear_output(workspace);
  if (anchors == nullptr || anchor_count == nullptr) return;
  const std::uint32_t count = anchor_count[0];
  if (count == 0u || count > surface_organ::kSurfaceOrganMaxAnchors) return;
  const surface_organ::OpaqueContentPlanView plan{anchors, count};
  surface_organ::realize_construction_witness(units, plan, witness, workspace);
}
