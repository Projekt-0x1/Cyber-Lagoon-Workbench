#!/usr/bin/env python3
"""Byte-blind resident arbitration followed by a separate learned voicebox."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import reference_resident_authenticated_probe_v1 as probe
from reference_resident_authenticated_probe_v1 import (
    _SURFACE_AUTHORITY, ProbeCommandV1, SurfaceApplyReceiptV1,
    SurfaceResponseContractV1,
)
from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1, _identity,
    _strict_numeric,
)
from reference_resident_partial_network_unfold_v1 import (
    PartialNetworkByteAncestryV1, _unfold_resident_partial_network_v1,
)
from reference_resident_surface_probe_catalyst_v1 import (
    ResidentSurfaceProbeCatalystV1,
)
from reference_resident_action_constructor_binding_v1 import (
    _BINDING_AUTHORITY, ActionConstructorBindingRefuse,
    ActionConstructorTrialV1, ResidentActionConstructorBindingV1,
)


MAX_CANDIDATES = 16
MAX_ROOTS = 1024


@dataclass(frozen=True)
class ResidentPartnerSurfaceSelectionV1:
    identity: int
    command_root: int
    mapping_root: int
    action_recipe: int
    constructor_root: int
    selection_state_root: int
    candidate_set_root: int
    surface_state_root: int
    surface_candidate_roots: tuple[int, ...]
    catalyst_recipe_roots: tuple[int, ...]
    catalyst_witness_roots: tuple[int, ...]
    binding_network_root: int
    binding_witness_roots: tuple[int, ...]
    binding_trial_roots: tuple[int, ...]
    binding_catalyst_witness_roots: tuple[int, ...]
    binding_trajectory_roots: tuple[int, ...]
    binding_trajectory_constituent_roots: tuple[int, ...]
    binding_surface_apply_roots: tuple[int, ...]
    nomination_root: int
    ticket_root: int
    ticket_envelope_root: int
    mapping_evidence_root: int
    probe_evidence_roots: tuple[int, ...]
    apply_receipt_roots: tuple[int, ...]
    return_contact_roots: tuple[int, ...]
    outcome_source_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    surface_work_units: int
    catalyst_work_units: int
    work_units: int


@dataclass(frozen=True)
class ResidentPartnerSurfaceByteAncestryV1:
    offset: int
    unit: int
    selection_root: int
    command_root: int
    catalyst_recipe_roots: tuple[int, ...]
    catalyst_witness_roots: tuple[int, ...]
    binding_network_root: int
    binding_witness_roots: tuple[int, ...]
    binding_trial_roots: tuple[int, ...]
    binding_catalyst_witness_roots: tuple[int, ...]
    binding_trajectory_roots: tuple[int, ...]
    binding_trajectory_constituent_roots: tuple[int, ...]
    binding_surface_apply_roots: tuple[int, ...]
    surface_candidate_roots: tuple[int, ...]
    nomination_root: int
    ticket_root: int
    ticket_envelope_root: int
    mapping_evidence_root: int
    probe_evidence_roots: tuple[int, ...]
    apply_receipt_roots: tuple[int, ...]
    return_contact_roots: tuple[int, ...]
    outcome_source_roots: tuple[int, ...]
    leaf: PartialNetworkByteAncestryV1


@dataclass(frozen=True)
class ResidentPartnerSurfaceTrajectoryV1:
    identity: int
    selection_root: int
    constructor_root: int
    units: tuple[int, ...]
    constituent_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    work_units: int
    ancestry: tuple[ResidentPartnerSurfaceByteAncestryV1, ...]


def _surface_candidates(resident):
    if not isinstance(resident, ResidentParametricSpanNetworkV1):
        raise ParametricSpanNetworkRefuse("partner_surface:surface_resident")
    base = resident._inner
    if base.pending or not base.samples:
        raise ParametricSpanNetworkRefuse("partner_surface:surface_frontier")
    work = 0
    def charge(amount=1):
        nonlocal work
        work += int(amount)
        if work > resident.work_limit:
            raise ParametricSpanNetworkRefuse("partner_surface:resource")
    charge(len(base.samples) + len(base.span_occurrences) + len(base.recipes))
    newest = base.samples[-1]
    ordered = [row for row in base.samples
               if row.source == newest.source and row.channel == newest.channel]
    position = {row.identity: index for index, row in enumerate(ordered)}
    recipes = {(row.channel, row.length, row.span_hash): row
               for row in base.recipes.values()}
    frontier = []
    for occurrence in base.span_occurrences:
        charge()
        if (occurrence.source != newest.source or occurrence.channel != newest.channel
                or not occurrence.sample_roots):
            continue
        recipe = recipes.get((occurrence.channel, occurrence.length,
                              occurrence.span_hash))
        if recipe is not None:
            frontier.append((occurrence, recipe))
    right = [(occurrence, recipe) for occurrence, recipe in frontier
             if occurrence.sample_roots[-1] == newest.identity]
    rows = set()
    current_sources = set()
    for right_occurrence, right_recipe in right:
        right_start = position.get(right_occurrence.sample_roots[0], -1)
        for left_occurrence, left_recipe in frontier:
            charge()
            if position.get(left_occurrence.sample_roots[-1], -3) + 1 != right_start:
                continue
            pair = (left_recipe.identity, right_recipe.identity)
            for constructor in resident.constructors.values():
                charge(4 + len(constructor.binding_pairs)
                       + len(constructor.binding_pair_sources)
                       + len(constructor.binding_pairs) ** 2
                       + sum(len(row[2])
                             for row in constructor.binding_pair_sources))
                pair_sources = {row[0:2]: row[2]
                                for row in constructor.binding_pair_sources}
                if (constructor.middle_recipe
                        and left_recipe.identity in constructor.left_bindings
                        and right_recipe.identity in constructor.right_bindings
                        and pair not in constructor.binding_pairs
                        and resident._three_corner_support(
                            set(constructor.binding_pairs), *pair)
                        and resident._cross_source_three_corner(pair_sources, *pair)):
                    rows.add(constructor.identity)
                    current_sources.update((left_occurrence.source,
                                            right_occurrence.source))
    candidates = tuple(sorted(rows))
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ParametricSpanNetworkRefuse("partner_surface:candidate_ambiguity")
    state_values = (
        resident.session, resident._inner.next_sequence, newest.identity,
        tuple(sorted(resident.constructors)), candidates,
        tuple(sorted(resident._inner.withdrawn_sources)), resident.work_limit,
    )
    return candidates, _identity(b"resident-partner-surface-state-v1", state_values), \
        tuple(sorted(current_sources)), work


def _current_command_context(catalyst):
    if not isinstance(catalyst, ResidentSurfaceProbeCatalystV1):
        raise ParametricSpanNetworkRefuse("partner_surface:catalyst_resident")
    pending = catalyst.pending
    if pending is None or pending.apply_receipt is not None:
        raise ParametricSpanNetworkRefuse("partner_surface:pending")
    command = pending.command
    ticket = pending.ticket
    if (not isinstance(command, ProbeCommandV1)
            or command.identity != probe._identity(
                b"authenticated-probe-command-v1", command.content_fields())
            or command != catalyst.pending.command
            or command.mapping_root != catalyst.mapping.identity
            or not catalyst._body_boundary.valid_mapping(catalyst.mapping)
            or (command.action_recipe, command.command_value)
            not in catalyst.mapping.entries
            or (command.ticket, command.ticket_envelope_root, command.session,
                command.incarnation, command.deadline_sequence)
            != (ticket.ticket, ticket.envelope_root, ticket.session,
                ticket.incarnation, ticket.deadline_sequence)
            or pending.episode != command.episode
            or catalyst._inner.pending.get(ticket.ticket) != ticket
            or not command.opened_sequence <= catalyst.next_sequence
            <= command.deadline_sequence
            or catalyst.resident_tick > command.deadline_resident_tick):
        raise ParametricSpanNetworkRefuse("partner_surface:command_authentication")
    nominations = [row for row in catalyst.context_nominations
                   if row.command == command.identity
                   and row.mapping_root == catalyst.mapping.identity]
    if len(nominations) != 1:
        raise ParametricSpanNetworkRefuse("partner_surface:context_currentness")
    return command, ticket, nominations[0]


def nominate_resident_action_constructor_trial_v1(
        catalyst: ResidentSurfaceProbeCatalystV1,
        surface: ResidentParametricSpanNetworkV1,
        binding: ResidentActionConstructorBindingV1) -> ActionConstructorTrialV1:
    command, _ticket, nomination = _current_command_context(catalyst)
    candidates, surface_state, current_sources, _work = _surface_candidates(surface)
    return binding._nominate(command.identity, command.action_recipe,
        nomination.identity, candidates, surface_state,
        tuple(sorted({*current_sources, *nomination.source_roots})),
        command.deadline_resident_tick, _BINDING_AUTHORITY)


def unfold_resident_action_constructor_trial_v1(
        binding: ResidentActionConstructorBindingV1,
        surface: ResidentParametricSpanNetworkV1,
        trial: ActionConstructorTrialV1):
    if (not isinstance(binding, ResidentActionConstructorBindingV1)
            or binding.pending is None or binding.pending.trial != trial):
        raise ParametricSpanNetworkRefuse("partner_surface:binding_trial")
    candidates, state_root, _sources, work = _surface_candidates(surface)
    if (candidates != trial.candidate_roots or state_root != trial.surface_state_root
            or work >= surface.work_limit):
        raise ParametricSpanNetworkRefuse("partner_surface:binding_frontier")
    old_limit = surface.work_limit
    try:
        surface.work_limit = old_limit - work
        trajectory = _unfold_resident_partial_network_v1(
            surface, trial.constructor_root)
    finally:
        surface.work_limit = old_limit
    binding._mark_trajectory(trial, trajectory.identity,
        trajectory.constituent_roots, _BINDING_AUTHORITY)
    return trajectory


def bind_resident_action_constructor_surface_v1(
        catalyst: ResidentSurfaceProbeCatalystV1,
        surface: ResidentParametricSpanNetworkV1,
        binding: ResidentActionConstructorBindingV1,
        contract: SurfaceResponseContractV1,
        trial: ActionConstructorTrialV1,
        trajectory):
    """Apply the chosen voicebox trajectory to the external response boundary."""
    expected = _unfold_resident_partial_network_v1(
        surface, trial.constructor_root)
    if (not isinstance(binding, ResidentActionConstructorBindingV1)
            or binding.pending is None or binding.pending.trial != trial
            or binding.pending.trajectory_root != int(trajectory.identity)
            or trajectory != expected
            or catalyst.pending is None
            or catalyst.pending.command.identity != trial.command_root):
        raise ParametricSpanNetworkRefuse("partner_surface:surface_apply_state")
    catalyst_before = catalyst._snapshot()
    body_before = catalyst._body_boundary._snapshot_state()
    binding_before = binding._snapshot()
    try:
        apply = catalyst.dispatch()
        receipt = catalyst._body_boundary._bind_resident_surface(
            catalyst.mapping, contract, apply, trial, trajectory,
            _SURFACE_AUTHORITY)
        fields = (*receipt.signed_fields(), receipt.auth_tag)
        binding._mark_surface_apply(trial, fields, _BINDING_AUTHORITY)
        return apply, receipt
    except Exception:
        catalyst._body_boundary._restore_state(body_before)
        catalyst._rollback(catalyst_before)
        binding._rollback(binding_before)
        raise


def settle_resident_action_constructor_trial_v1(
        catalyst: ResidentSurfaceProbeCatalystV1,
        binding: ResidentActionConstructorBindingV1):
    if (not isinstance(binding, ResidentActionConstructorBindingV1)
            or binding.pending is None or catalyst.pending is not None):
        raise ParametricSpanNetworkRefuse("partner_surface:binding_settlement")
    catalyst._validate_catalyst_state(); trial = binding.pending.trial
    rows = [row for row in catalyst.catalyst_witnesses
            if row.nomination == trial.context_nomination_root
            and row.action_recipe == trial.action_recipe]
    nomination_by_root = {row.identity: row
        for row in catalyst.context_nominations}
    nomination = nomination_by_root.get(trial.context_nomination_root)
    if (len(rows) != 1 or not catalyst.catalyst_witnesses
            or rows[0] != catalyst.catalyst_witnesses[-1]
            or not catalyst.context_nominations
            or trial.context_nomination_root
                != catalyst.context_nominations[-1].identity
            or nomination is None or nomination.command != trial.command_root
            or catalyst.resident_tick > trial.deadline_tick
            or len(rows[0].probe_evidence_fields) != 10
            or rows[0].probe_evidence_fields[2] != trial.command_root):
        raise ParametricSpanNetworkRefuse("partner_surface:binding_evidence")
    row = rows[0]
    fields = binding.pending.surface_apply_fields
    try:
        surface_apply = SurfaceApplyReceiptV1(*fields)
    except TypeError as exc:
        raise ParametricSpanNetworkRefuse(
            "partner_surface:surface_apply_evidence") from exc
    returns = []
    for event in catalyst.events:
        if event.kind != probe.EVENT_RETURN:
            continue
        signed, tag = event.values
        occurrence = probe.OccurrenceContactV1(*signed[8], signed[9])
        returned = probe.BodyReturnContactV1(*signed[:8], occurrence, tag)
        if returned.command_hash == trial.command_root:
            returns.append(returned)
    if (not catalyst._body_boundary.valid_surface_apply_evidence(surface_apply)
            or surface_apply.trial_root != trial.identity
            or surface_apply.trajectory_root != binding.pending.trajectory_root
            or surface_apply.apply_receipt != row.probe_evidence_fields[4]
            or not returns
            or any(surface_apply.identity not in returned.occurrence.provenance
                   for returned in returns)):
        raise ParametricSpanNetworkRefuse(
            "partner_surface:surface_apply_evidence")
    return binding._settle(trial, row.identity, row.probe_evidence, row.difference,
        row.outcome_source, row.source_roots, _BINDING_AUTHORITY)


def select_resident_partner_surface_v1(
        catalyst: ResidentSurfaceProbeCatalystV1,
        surface: ResidentParametricSpanNetworkV1,
        binding: ResidentActionConstructorBindingV1
        ) -> ResidentPartnerSurfaceSelectionV1:
    """Transform bounded resident state into a numeric choice; emit no units."""
    command, ticket, nomination = _current_command_context(catalyst)
    candidates, surface_state, current_sources, work = _surface_candidates(surface)
    catalyst_work = (len(catalyst.context_nominations) * 4
        + len(catalyst.catalyst_recipes) * 8
        + len(catalyst.catalyst_witnesses) * 16
        + len(catalyst.catalyst_witnesses) ** 2
        + len(catalyst.events) * 16
        + sum(len(row.source_roots) for row in catalyst.catalyst_witnesses)
        + 1)
    if catalyst_work > catalyst.work_limit:
        raise ParametricSpanNetworkRefuse("partner_surface:catalyst_resource")
    def charge_catalyst(amount=1):
        nonlocal catalyst_work
        catalyst_work += int(amount)
        if catalyst_work > catalyst.work_limit:
            raise ParametricSpanNetworkRefuse("partner_surface:catalyst_resource")
    catalyst._validate_catalyst_state()
    live_witnesses = {row.identity: row for row in catalyst.catalyst_witnesses}
    selected_recipes = []
    for row in catalyst.catalyst_recipes.values():
        witnesses = [live_witnesses.get(root) for root in row.witness_roots]
        if (row.span_recipe == nomination.span_recipe
                and row.mapping_root == catalyst.mapping.identity
                and witnesses and all(item is not None
                    and item.action_recipe == command.action_recipe
                    for item in witnesses)):
            selected_recipes.append(row)
    if len(selected_recipes) != 1:
        raise ParametricSpanNetworkRefuse("partner_surface:catalyst_relation")
    selected_recipe = selected_recipes[0]
    recipes = (selected_recipe.identity,)
    witness_roots = selected_recipe.witness_roots
    witnesses = tuple(live_witnesses[root] for root in witness_roots)
    if (selected_recipe.credit <= 0
            or len({row.outcome_source for row in witnesses
                    if row.difference > 0}) < 2
            or sum(row.difference for row in witnesses) != selected_recipe.credit
            or any(row.identity != probe._identity(
                b"surface-probe-catalyst-witness-v1", (row.nomination,
                    row.span_recipe, row.action_recipe, row.mapping_root,
                    row.probe_evidence, row.probe_evidence_fields,
                    row.outcome_source, row.difference, row.source_roots))
                or row.probe_evidence != probe._identity(
                    b"authenticated-probe-evidence-v1",
                    row.probe_evidence_fields) for row in witnesses)):
        raise ParametricSpanNetworkRefuse("partner_surface:catalyst_currentness")
    nomination_by_root = {row.identity: row for row in catalyst.context_nominations}
    consequence_commands = {nomination_by_root[row.nomination].command
                            for row in witnesses}
    applies = []
    returns = []
    for event in catalyst.events:
        if event.kind == probe.EVENT_APPLY:
            signed, tag = event.values
            receipt = probe.BodyApplyReceiptV1(*signed, tag)
            if receipt.command_hash in consequence_commands:
                applies.append(receipt)
        elif event.kind == probe.EVENT_RETURN:
            signed, tag = event.values
            occurrence = probe.OccurrenceContactV1(*signed[8], signed[9])
            returned = probe.BodyReturnContactV1(*signed[:8], occurrence, tag)
            if returned.command_hash in consequence_commands:
                returns.append(returned)
    apply_by_command = {row.command_hash: row for row in applies}
    returns_by_command = {root: [] for root in consequence_commands}
    for row in returns:
        returns_by_command[row.command_hash].append(row)
    evidence_current = True
    charge_catalyst(len(catalyst._inner.witnesses)
                    + len(catalyst._inner.actual_nodes)
                    + len(catalyst._inner.nominations))
    structural_by_root = {row.identity: row for row in catalyst._inner.witnesses}
    node_by_root = {row.identity: row for row in catalyst._inner.actual_nodes}
    inner_nomination_roots = {row.identity for row in catalyst._inner.nominations}
    for row in witnesses:
        nomination_row = nomination_by_root[row.nomination]
        evidence = row.probe_evidence_fields
        returned = returns_by_command.get(nomination_row.command, ())
        apply = apply_by_command.get(nomination_row.command)
        structural = structural_by_root.get(evidence[6]) if len(evidence) == 10 else None
        node = (None if structural is None
                else node_by_root.get(structural.observed_node))
        if structural is not None:
            charge_catalyst(16 + len(structural.relation_recipe_roots) * 4
                + len(structural.hypothesis_roots)
                + len(structural.nomination_roots)
                + len(structural.source_roots)
                + len(structural.ancestry) * 24)
        relation_recipes = ([] if structural is None else
            [catalyst._inner.recipes.get(root)
             for root in structural.relation_recipe_roots])
        charge_catalyst(sum(len(recipe.witness_roots)
            + len(recipe.source_roots) for recipe in relation_recipes
            if recipe is not None))
        relation_witness_roots = tuple(sorted({root
            for recipe in relation_recipes if recipe is not None
            for root in recipe.witness_roots}))
        relation_source_roots = tuple(sorted({source
            for recipe in relation_recipes if recipe is not None
            for source in recipe.source_roots}))
        structural_body = (() if structural is None or node is None else (
            structural.ticket_envelope_root, node.identity,
            node.evidence_revision, node.source, node.source_roots,
            node.ancestry_roots, structural.difference,
            structural.marginal_recipe, structural.hypothesis_roots,
            structural.nomination_roots, structural.relation_recipe_roots,
            relation_witness_roots, relation_source_roots,
            structural.source_roots,
            tuple(asdict(item) for item in structural.ancestry)))
        evidence_current = evidence_current and (
            isinstance(evidence, tuple) and len(evidence) == 10
            and evidence[0] == nomination_row.command_fields[0]
            and evidence[1] == row.action_recipe
            and evidence[2] == nomination_row.command
            and evidence[3] == row.mapping_root
            and apply is not None and evidence[4] == apply.identity
            and structural is not None and node is not None
            and evidence[5] == structural.ticket
            and evidence[6] == structural.identity
            and structural.identity == _identity(
                b"composite-cue-prediction-witness-body-v1", structural_body)
            and len(relation_recipes) == len(structural.relation_recipe_roots)
            and all(item is not None for item in relation_recipes)
            and set(structural.hypothesis_roots).issubset(catalyst._inner.hypotheses)
            and set(structural.nomination_roots).issubset(inner_nomination_roots)
            and evidence[7] == row.difference
            and evidence[8] == tuple(item.identity for item in returned)
            and returned
            and returned[-1].occurrence.source == row.outcome_source
            and structural.source == row.outcome_source
            and evidence[9] == structural.source_roots
            and row.outcome_source in evidence[9])
    if (set(apply_by_command) != consequence_commands
            or not evidence_current
            or any(not catalyst._body_boundary.valid_apply(row)
                   for row in applies)
            or any(not catalyst._body_boundary.valid_return(row)
                   or row.apply_receipt != apply_by_command[row.command_hash].identity
                   for row in returns)
            or {row.outcome_source for row in witnesses}
            != {row.occurrence.source for row in returns}):
        raise ParametricSpanNetworkRefuse("partner_surface:consequence_lineage")
    probe_evidence_roots = tuple(sorted(row.probe_evidence for row in witnesses))
    apply_roots = tuple(sorted(row.identity for row in applies))
    return_roots = tuple(sorted(row.identity for row in returns))
    outcome_sources = tuple(sorted(row.outcome_source for row in witnesses))
    try:
        binding_network = binding.resolve(
            command.action_recipe, candidates, _BINDING_AUTHORITY)
    except ActionConstructorBindingRefuse as exc:
        if "resource" in str(exc):
            raise ParametricSpanNetworkRefuse(
                "partner_surface:binding_resource") from exc
        raise ParametricSpanNetworkRefuse("partner_surface:binding_unresolved") from exc
    constructor = surface.constructors.get(binding_network.constructor_root)
    if constructor is None:
        raise ParametricSpanNetworkRefuse("partner_surface:binding_constructor")
    binding._bounded()
    binding_witness_by_root = {row.identity: row for row in binding.witnesses}
    binding_trial_by_root = {row.identity: row for row in binding.trials}
    binding_witnesses = tuple(binding_witness_by_root.get(root)
        for root in binding_network.witness_roots)
    binding_trials = tuple(None if row is None else
        binding_trial_by_root.get(row.trial_root) for row in binding_witnesses)
    binding_catalyst_witnesses = tuple(None if row is None else
        live_witnesses.get(row.catalyst_witness_root) for row in binding_witnesses)
    try:
        binding_surface_applies = tuple(SurfaceApplyReceiptV1(
            *row.surface_apply_fields) for row in binding_witnesses
            if row is not None)
    except TypeError as exc:
        raise ParametricSpanNetworkRefuse("partner_surface:binding_lineage") from exc
    if (any(row is None for row in binding_witnesses)
            or any(row is None for row in binding_trials)
            or any(row is None for row in binding_catalyst_witnesses)
            or any(catalyst_row.nomination not in nomination_by_root
                or trial.context_nomination_root != catalyst_row.nomination
                or trial.command_root != nomination_by_root[catalyst_row.nomination].command
                or trial.action_recipe != catalyst_row.action_recipe
                or witness.probe_evidence_root != catalyst_row.probe_evidence
                or witness.difference != catalyst_row.difference
                or witness.outcome_source != catalyst_row.outcome_source
                or witness.catalyst_witness_root != catalyst_row.identity
                or not catalyst._body_boundary.valid_surface_apply_evidence(surface_apply)
                or surface_apply.trial_root != trial.identity
                or surface_apply.trajectory_root != witness.trajectory_root
                or surface_apply.apply_receipt != catalyst_row.probe_evidence_fields[4]
                or not any(surface_apply.identity
                    in returned.occurrence.provenance
                    for returned in returns_by_command.get(trial.command_root, ()))
                for witness, trial, catalyst_row, surface_apply in zip(
                    binding_witnesses, binding_trials,
                    binding_catalyst_witnesses, binding_surface_applies))):
        raise ParametricSpanNetworkRefuse("partner_surface:binding_lineage")
    binding_trial_roots = tuple(row.identity for row in binding_trials)
    binding_catalyst_roots = tuple(row.identity
        for row in binding_catalyst_witnesses)
    binding_trajectory_roots = tuple(row.trajectory_root
        for row in binding_witnesses)
    binding_trajectory_constituents = tuple(sorted({root
        for row in binding_witnesses
        for root in row.trajectory_constituent_roots}))
    binding_surface_apply_roots = tuple(row.identity
        for row in binding_surface_applies)
    sources = tuple(sorted({*current_sources, *nomination.source_roots,
        *constructor.source_roots, *binding_network.source_roots,
        *(source for root in witness_roots
          for source in live_witnesses[root].source_roots)}))
    work += len(recipes) + len(witness_roots) + len(sources) + len(candidates)
    if (work > surface.work_limit or catalyst_work > catalyst.work_limit
            or len(sources) > MAX_ROOTS
            or any(source in surface._inner.withdrawn_sources for source in sources)):
        raise ParametricSpanNetworkRefuse("partner_surface:resource")
    values = (command.identity, command.mapping_root, command.action_recipe,
        constructor.identity, command.selection_state_root,
        command.candidate_set_root, surface_state, candidates, recipes,
        witness_roots, binding_network.identity,
        binding_network.witness_roots, binding_trial_roots,
        binding_catalyst_roots, binding_trajectory_roots,
        binding_trajectory_constituents, binding_surface_apply_roots,
        nomination.identity,
        ticket.ticket, ticket.envelope_root,
        catalyst.mapping.route_root, probe_evidence_roots, apply_roots,
        return_roots, outcome_sources, sources, work, catalyst_work,
        work + catalyst_work)
    result = ResidentPartnerSurfaceSelectionV1(
        _identity(b"resident-partner-surface-selection-v1", values), *values)
    _strict_numeric(result, extent=MAX_ROOTS)
    return result


def unfold_resident_partner_surface_v1(
        catalyst: ResidentSurfaceProbeCatalystV1,
        surface: ResidentParametricSpanNetworkV1,
        binding: ResidentActionConstructorBindingV1,
        selection: ResidentPartnerSurfaceSelectionV1
        ) -> ResidentPartnerSurfaceTrajectoryV1:
    """Voicebox: authenticate a resident choice, then materialize its Network."""
    if (not isinstance(selection, ResidentPartnerSurfaceSelectionV1)
            or selection != select_resident_partner_surface_v1(
                catalyst, surface, binding)):
        raise ParametricSpanNetworkRefuse("partner_surface:selection_authentication")
    remaining = surface.work_limit - selection.surface_work_units
    if remaining <= 0:
        raise ParametricSpanNetworkRefuse("partner_surface:resource")
    old_limit = surface.work_limit
    try:
        surface.work_limit = remaining
        inner = _unfold_resident_partial_network_v1(
            surface, selection.constructor_root)
    finally:
        surface.work_limit = old_limit
    surface_work = selection.surface_work_units + inner.work_units
    work = selection.catalyst_work_units + surface_work
    if surface_work > old_limit:
        raise ParametricSpanNetworkRefuse("partner_surface:resource")
    ancestry = tuple(ResidentPartnerSurfaceByteAncestryV1(
        row.offset, row.unit, selection.identity, selection.command_root,
        selection.catalyst_recipe_roots, selection.catalyst_witness_roots,
        selection.binding_network_root, selection.binding_witness_roots,
        selection.binding_trial_roots,
        selection.binding_catalyst_witness_roots,
        selection.binding_trajectory_roots,
        selection.binding_trajectory_constituent_roots,
        selection.binding_surface_apply_roots,
        selection.surface_candidate_roots, selection.nomination_root,
        selection.ticket_root, selection.ticket_envelope_root,
        selection.mapping_evidence_root, selection.probe_evidence_roots,
        selection.apply_receipt_roots, selection.return_contact_roots,
        selection.outcome_source_roots, row) for row in inner.ancestry)
    roots = tuple(sorted({selection.identity, selection.command_root,
        selection.mapping_root, selection.surface_state_root,
        *selection.surface_candidate_roots, *selection.catalyst_recipe_roots,
        *selection.catalyst_witness_roots, selection.binding_network_root,
        *selection.binding_witness_roots, *selection.binding_trial_roots,
        *selection.binding_catalyst_witness_roots,
        *selection.binding_trajectory_roots,
        *selection.binding_trajectory_constituent_roots,
        *selection.binding_surface_apply_roots,
        selection.nomination_root,
        selection.ticket_root, selection.ticket_envelope_root,
        selection.mapping_evidence_root, *selection.probe_evidence_roots,
        *selection.apply_receipt_roots, *selection.return_contact_roots,
        *inner.constituent_roots}))
    sources = tuple(sorted({*selection.source_roots, *inner.source_roots}))
    ancestry_values = tuple((row.offset, row.unit, row.selection_root,
        row.command_root, row.catalyst_recipe_roots,
        row.catalyst_witness_roots, row.binding_network_root,
        row.binding_witness_roots, row.binding_trial_roots,
        row.binding_catalyst_witness_roots, row.binding_trajectory_roots,
        row.binding_trajectory_constituent_roots,
        row.binding_surface_apply_roots,
        row.surface_candidate_roots,
        row.nomination_root, row.ticket_root, row.ticket_envelope_root,
        row.mapping_evidence_root, row.probe_evidence_roots,
        row.apply_receipt_roots, row.return_contact_roots,
        row.outcome_source_roots,
        tuple(getattr(row.leaf, name) for name in row.leaf.__dataclass_fields__))
        for row in ancestry)
    values = (selection.identity, inner.constructor_root, inner.units, roots,
              sources, work, ancestry_values)
    result = ResidentPartnerSurfaceTrajectoryV1(
        _identity(b"resident-partner-surface-trajectory-v1", values),
        selection.identity, inner.constructor_root, inner.units, roots,
        sources, work, ancestry)
    _strict_numeric(result, extent=MAX_ROOTS)
    return result
