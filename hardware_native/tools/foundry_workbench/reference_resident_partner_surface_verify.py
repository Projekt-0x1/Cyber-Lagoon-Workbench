#!/usr/bin/env python3
"""Fast hostile assay for consequence-conditioned resident surface choice."""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

import reference_resident_authenticated_probe_v1 as probe
from reference_resident_authenticated_probe_v1 import (
    ACTION_RECIPES, admit_body_probe_boundary_v1,
)
from reference_resident_channel_sequence_grounding_v1 import (
    admit_channel_sequence_boundary_v1,
)
from reference_resident_composite_cue_prediction_verify import main as trained_fixture
from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1,
)
from reference_resident_parametric_span_network_verify import acquire, arrange, put
from reference_resident_partial_network_unfold_v1 import (
    unfold_resident_partial_network_v1,
)
from reference_resident_partner_surface_v1 import (
    ResidentPartnerSurfaceSelectionV1,
    _current_command_context,
    _surface_candidates,
    bind_resident_action_constructor_surface_v1,
    nominate_resident_action_constructor_trial_v1,
    select_resident_partner_surface_v1,
    settle_resident_action_constructor_trial_v1,
    unfold_resident_action_constructor_trial_v1,
    unfold_resident_partner_surface_v1,
)
from reference_resident_action_constructor_binding_v1 import (
    _BINDING_AUTHORITY,
    ResidentActionConstructorBindingV1,
    admit_action_constructor_checkpoint_boundary_v1,
)
from reference_resident_surface_probe_catalyst_v1 import (
    ResidentSurfaceProbeCatalystV1,
)


CLAIM = "ACQUIRED_PAIR_REAFFERENT_ACTION_NETWORK_BINDS_VOICEBOX_REFERENCE_ONLY"
STARTED = time.perf_counter()


def b(value): return tuple(value.encode("utf-8"))


ALICE, CAROL, BOB, DAVE = map(b, ("Ari", "Cia", "Bob", "Dan"))
REGARDS, GREETS = map(b, (" likes ", " sees "))


def refuses(call, fragment=""):
    try:
        call()
    except (RuntimeError, TypeError, ValueError) as exc:
        return fragment in str(exc)
    return False


def feed(machine, boundary, source, *spans):
    for span in spans:
        for unit in span:
            put(boundary, machine, source, unit)


def contains(machine, needle):
    histories = {}
    for row in machine._inner.samples:
        histories.setdefault((row.source, row.channel), []).append(row.unit)
    return any(any(tuple(units[index:index + len(needle)]) == needle
                   for index in range(len(units) - len(needle) + 1))
               for units in histories.values())


def assay(ctx):
    checks = dict(ctx["checks"])
    contact_boundary, trained = ctx["boundary"], ctx["machine"]
    world, consequence, distractor = ctx["world"], ctx["consequence"], ctx["distractor"]
    learned = ctx["learned"]
    outer = sorted(span for channel, span in learned if channel == 7 and span[0] < 200)
    context_span, novel_span = outer[:2]

    surface_boundary = admit_channel_sequence_boundary_v1()
    surface = ResidentParametricSpanNetworkV1(surface_boundary)
    for index, span in enumerate((ALICE, CAROL, BOB, DAVE, REGARDS, GREETS)):
        acquire(surface_boundary, surface, span, 180 + index)
    for middle, base_source in ((REGARDS, 3100), (GREETS, 3200)):
        for offset, left, right in ((1, ALICE, BOB), (2, ALICE, BOB),
                                    (1, ALICE, DAVE), (2, CAROL, DAVE)):
            arrange(surface_boundary, surface, base_source + offset,
                    left, middle, right)
    middle_roots = {}
    for span in (REGARDS, GREETS):
        middle_roots[span] = next(row.identity for row in surface._inner.recipes.values()
            if any(surface._inner._units(occurrence) == span
                and (occurrence.channel, occurrence.length, occurrence.span_hash)
                == (row.channel, row.length, row.span_hash)
                for occurrence in surface._inner.span_occurrences))
    constructors = {row.middle_recipe: row for row in surface.constructors.values()}
    checks["two_alternative_compact_networks_learn_same_slot_frontier"] = (
        set(constructors) == set(middle_roots.values())
        and constructors[middle_roots[REGARDS]].binding_pairs
        == constructors[middle_roots[GREETS]].binding_pairs)
    regards_root = constructors[middle_roots[REGARDS]].identity
    greets_root = constructors[middle_roots[GREETS]].identity
    heldout = (CAROL + REGARDS + BOB, CAROL + GREETS + BOB)
    checks["both_complete_heldout_surfaces_absent_from_contacts"] = (
        not any(contains(surface, row) for row in heldout))
    feed(surface, surface_boundary, 3301, CAROL, BOB)
    checks["voicebox_without_resident_choice_refuses_real_ambiguity"] = refuses(
        lambda: unfold_resident_partial_network_v1(surface), "ambiguous")
    def surface_state(machine):
        return (tuple(machine.constructors.items()), tuple(machine.networks.items()),
                tuple(machine.rebinds), tuple(machine.events), tuple(machine.trace),
                tuple(machine._inner.samples), tuple(machine._inner.span_occurrences),
                tuple(machine._inner.recipes.items()),
                tuple(machine._inner.prediction_witnesses), machine.work,
                machine.work_limit)

    def catalyst_state(machine):
        return (machine.pending, tuple(machine.events), tuple(machine.evidence),
                tuple(machine.context_nominations), tuple(machine.catalyst_witnesses),
                tuple(machine.catalyst_recipes.items()), machine.resource,
                machine.resident_tick, machine.work)

    def binding_state(machine):
        return (machine.pending, tuple(machine.trials), tuple(machine.witnesses),
                tuple(machine.networks.items()), tuple(sorted(machine.withdrawn_sources)),
                machine.tick, machine.work, machine.work_limit)

    body = admit_body_probe_boundary_v1(contact_boundary)
    mapping = body.precommit(trained.session, 191, 51, 3, (
        (ACTION_RECIPES[0].identity, 9101,
         (51001, 51002, 51003, 51004), 9, distractor),
        (ACTION_RECIPES[1].identity, 9102,
         (52001, 52002, 52003, 52004), 9, consequence),
    ))
    ordered_constructors = tuple(sorted((regards_root, greets_root)))
    initial_index = probe._identity(
        b"action-constructor-endogenous-exploration-v1",
        (ACTION_RECIPES[1].identity,
         ResidentActionConstructorBindingV1._candidate_root(
             ordered_constructors))) % 2
    positive_constructor = ordered_constructors[1 - initial_index]
    positive_units = (heldout[0] if positive_constructor == regards_root
                      else heldout[1])
    negative_units = (heldout[1] if positive_constructor == regards_root
                      else heldout[0])
    response_contract = body.precommit_surface_responses(mapping, (
        (ACTION_RECIPES[0].identity, body.surface_unit_digest(positive_units), 8101,
         (51001, 51002, 51003, 51004), 9, distractor),
        (ACTION_RECIPES[0].identity, body.surface_unit_digest(negative_units), 8102,
         (51001, 51002, 51003, 51004), 9, distractor),
        (ACTION_RECIPES[1].identity, body.surface_unit_digest(positive_units), 8103,
         (52001, 52002, 52003, 52004), 9, consequence),
        (ACTION_RECIPES[1].identity, body.surface_unit_digest(negative_units), 8104,
         (51001, 51002, 51003, 51004), 9, distractor),
    ))
    catalyst = ResidentSurfaceProbeCatalystV1(
        contact_boundary, body, mapping,
        initial_inner=probe._unpack_inner(probe._pack_inner(trained), contact_boundary))
    binding_boundary = admit_action_constructor_checkpoint_boundary_v1()
    binding = ResidentActionConstructorBindingV1(binding_boundary)

    def contact(machine, source, channel, unit):
        row = contact_boundary.seal_sample(
            machine.session, machine.next_sequence, source, channel, (unit,),
            (source, channel))
        return machine.ingest_sample(row)

    def open_episode(machine, source, span):
        for unit in span:
            contact(machine, source, 7, unit)
        for unit in world:
            contact(machine, source, 8, unit)
        result = machine.tick()
        assert result.command is not None
        return result.command

    def settle(machine, apply=None):
        apply = machine.dispatch() if apply is None else apply
        evidence = None
        while machine.pending is not None:
            returned = body.seal_return(machine.mapping, apply, machine.next_sequence)
            evidence = machine.ingest_return(returned)
        return evidence

    commands, differences, trial_outputs = [], [], []
    surface_applies = []
    forged_trajectory_refused = False
    cross_component_rollback = False
    cross_refused = False
    cross_catalyst = cross_binding = cross_body = False
    for source in (60001, 60002, 60003, 60004, 60005, 60006):
        command = open_episode(catalyst, source, context_span)
        trial = nominate_resident_action_constructor_trial_v1(
            catalyst, surface, binding)
        trajectory = unfold_resident_action_constructor_trial_v1(
            binding, surface, trial)
        trial_outputs.append(trajectory)
        if not surface_applies:
            before_forgery = (catalyst_state(catalyst), binding_state(binding),
                              body._snapshot_state())
            forged_trajectory_refused = (
                refuses(lambda: bind_resident_action_constructor_surface_v1(
                    catalyst, surface, binding, response_contract, trial,
                    replace(trajectory, units=tuple(reversed(trajectory.units)))),
                    "surface_apply_state")
                and before_forgery == (catalyst_state(catalyst),
                    binding_state(binding), body._snapshot_state()))
            low_pending = ResidentActionConstructorBindingV1.restore(
                binding.checkpoint(), binding_boundary)
            low_pending.work_limit = 1; low_pending.work = 0
            before_low = (catalyst.checkpoint(),
                binding_state(low_pending), body._snapshot_state())
            cross_refused = refuses(lambda:
                bind_resident_action_constructor_surface_v1(
                    catalyst, surface, low_pending, response_contract, trial,
                    trajectory))
            cross_catalyst = before_low[0] == catalyst.checkpoint()
            cross_binding = before_low[1] == binding_state(low_pending)
            cross_body = before_low[2] == body._snapshot_state()
            cross_component_rollback = cross_catalyst and cross_binding and cross_body
        apply, surface_apply = bind_resident_action_constructor_surface_v1(
            catalyst, surface, binding, response_contract, trial, trajectory)
        surface_applies.append(surface_apply)
        evidence = settle(catalyst, apply)
        settle_resident_action_constructor_trial_v1(catalyst, binding)
        commands.append(command)
        differences.append(evidence.difference)
        if binding.networks:
            break

    checks["authenticated_consequence_transitions_voice_a_minus_b_plus_b_plus"] = (
        len(commands) <= 6 and differences.count(1) >= 2
        and differences.count(-1) >= 1
        and all(body.valid_surface_apply(row) for row in surface_applies))
    checks["forged_same_identity_trajectory_refuses_atomically"] = (
        forged_trajectory_refused)
    checks["dispatch_surface_apply_and_binding_mark_rollback_together"] = (
        cross_component_rollback and cross_refused)
    checks["low_binding_surface_apply_path_refuses"] = cross_refused
    checks["low_binding_surface_apply_path_restores_exact_state"] = (
        cross_component_rollback)
    checks["low_binding_surface_apply_restores_catalyst"] = cross_catalyst
    checks["low_binding_surface_apply_restores_binding"] = cross_binding
    checks["low_binding_surface_apply_restores_body"] = cross_body
    learned_binding = next(iter(binding.networks.values()))
    promoted_binding_checkpoint = binding.checkpoint()
    learned_units = (heldout[0] if learned_binding.constructor_root == regards_root
                     else heldout[1])
    checks["action_constructor_network_is_acquired_from_two_positive_sources"] = (
        len(binding.networks) == 1
        and learned_binding.constructor_root == positive_constructor
        and learned_binding.support == 2 and learned_binding.credit == 2
        and len({row.outcome_source for row in binding.witnesses
                 if row.identity in learned_binding.witness_roots}) == 2)
    learned_binding_witnesses = tuple(row for row in binding.witnesses
        if row.identity in learned_binding.witness_roots)
    learned_binding_trials = tuple(row for row in binding.trials
        if row.identity in {witness.trial_root
                            for witness in learned_binding_witnesses})
    checks["binding_witnesses_reconstruct_executed_trial_and_catalyst_lineage"] = (
        len(learned_binding_witnesses) == len(learned_binding_trials) == 2
        and all(witness.trajectory_root
                and witness.trajectory_constituent_roots
                and any(row.identity == witness.catalyst_witness_root
                    and row.probe_evidence == witness.probe_evidence_root
                    and row.difference == witness.difference
                    and row.outcome_source == witness.outcome_source
                    for row in catalyst.catalyst_witnesses)
                and any(row.identity == witness.trial_root
                    and row.constructor_root == witness.constructor_root
                    and row.command_root == witness.command_root
                    for row in learned_binding_trials)
                for witness in learned_binding_witnesses))
    learned_action_witnesses = tuple(row for row in binding.witnesses
        if row.action_recipe == learned_binding.action_recipe)
    checks["same_action_surface_counterfactual_drives_credit_and_exploration"] = (
        {row.difference for row in learned_action_witnesses} == {-1, 1}
        and len({row.constructor_root for row in learned_action_witnesses}) == 2
        and len({row.surface_apply_fields[9]
                 for row in learned_action_witnesses}) == 2
        and all(row.constructor_root == learned_binding.constructor_root
                for row in learned_action_witnesses if row.difference > 0)
        and all(row.constructor_root != learned_binding.constructor_root
                for row in learned_action_witnesses if row.difference < 0))
    checks["negative_surface_contact_cannot_teach_positive_network"] = (
        not {row.identity for row in learned_action_witnesses
             if row.difference < 0}.intersection(learned_binding.witness_roots))
    checks["external_surface_response_contract_is_opaque_and_byte_free_publicly"] = (
        len(response_contract.entries) == 4
        and all(len(row) == 3 for row in response_contract.entries)
        and not any(hasattr(response_contract, name)
                    for name in ("units", "bytes", "expected", "semantic")))
    checks["external_body_recomputes_complete_historical_trajectory_evidence"] = (
        all(body.valid_surface_apply_evidence(row) for row in surface_applies)
        and len(body._snapshot_state()[7]) == len(surface_applies))
    legacy_body = admit_body_probe_boundary_v1(contact_boundary)
    legacy_body._restore_state((0, (), (), ()))
    checks["legacy_v3_four_part_body_state_restores_with_empty_surface_extension"] = (
        legacy_body._snapshot_state()[4:] == ((), (), (), (), 0))
    checks["trial_adapter_was_resident_owned_and_voicebox_was_executed"] = (
        len(trial_outputs) == len(commands)
        and all(row.constructor_root in (regards_root, greets_root)
                for row in trial_outputs)
        and not any(hasattr(row, "expected") for row in binding.witnesses))
    forbidden_binding = {"units", "bytes", "output", "expected", "semantic",
                         "prompt", "style", "persona", "motor", "route", "domain"}
    checks["persistent_binding_network_is_numeric_and_has_no_surface_payload"] = (
        not forbidden_binding.intersection(
            name.lower() for name in learned_binding.__dataclass_fields__)
        and learned_binding.constructor_root not in dict(mapping.entries).values()
        and tuple(inspect.signature(
            nominate_resident_action_constructor_trial_v1).parameters)
            == ("catalyst", "surface", "binding"))

    final_command = open_episode(catalyst, 60004, context_span)
    final_selection = select_resident_partner_surface_v1(catalyst, surface, binding)
    final_output = unfold_resident_partner_surface_v1(
        catalyst, surface, binding, final_selection)
    checks["promoted_partner_relation_is_load_bearing_in_numeric_selection"] = (
        final_command.action_recipe == ACTION_RECIPES[1].identity
        and final_selection.constructor_root == learned_binding.constructor_root
        and final_selection.binding_network_root == learned_binding.identity
        and final_selection.binding_witness_roots == learned_binding.witness_roots
        and set(final_selection.binding_trial_roots)
            == {row.identity for row in learned_binding_trials}
        and set(final_selection.binding_catalyst_witness_roots)
            == {row.catalyst_witness_root for row in learned_binding_witnesses}
        and set(final_selection.binding_trajectory_roots)
            == {row.trajectory_root for row in learned_binding_witnesses}
        and set(final_selection.binding_surface_apply_roots)
            == {row.surface_apply_fields[0]
                for row in learned_binding_witnesses}
        and final_selection.catalyst_recipe_roots
        and final_selection.catalyst_witness_roots
        and final_output.units == learned_units)
    sham = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    checks["binding_sham_replays_exact_selection"] = (
        select_resident_partner_surface_v1(catalyst, surface, sham)
        == final_selection)
    checks["checkpoint_signer_is_not_public_or_host_callable"] = (
        not hasattr(binding_boundary, "seal")
        and refuses(lambda: binding_boundary._seal({}, object()),
                    "checkpoint_authority"))
    lesioned = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    lesioned.networks.clear()
    checks["binding_network_lesion_restores_ambiguity"] = refuses(
        lambda: select_resident_partner_surface_v1(
            catalyst, surface, lesioned), "binding_unresolved")
    selected_witnesses = [row for row in catalyst.catalyst_witnesses
        if row.identity in final_selection.catalyst_witness_roots]
    checks["condensed_probe_evidence_is_recomputed_and_one_to_one"] = (
        set(final_selection.probe_evidence_roots)
        == {row.probe_evidence for row in selected_witnesses}
        and all(row.probe_evidence == probe._identity(
            b"authenticated-probe-evidence-v1", row.probe_evidence_fields)
            for row in selected_witnesses))
    forbidden = {"units", "bytes", "output", "expected", "semantic", "prompt",
                 "style", "persona", "motor", "route", "channel", "domain"}
    checks["adapter_token_is_strict_numeric_and_cannot_emit_surface_bytes"] = (
        not forbidden.intersection(row.name.lower()
                                   for row in fields(ResidentPartnerSurfaceSelectionV1))
        and not hasattr(final_selection, "units")
        and tuple(inspect.signature(select_resident_partner_surface_v1).parameters)
        == ("catalyst", "surface", "binding"))
    checks["voicebox_has_exact_per_byte_selection_and_contact_ancestry"] = (
        len(final_output.ancestry) == len(final_output.units)
        and tuple(row.offset for row in final_output.ancestry)
        == tuple(range(len(final_output.units)))
        and all(row.unit == unit and row.selection_root == final_selection.identity
                and row.command_root == final_command.identity
                and row.binding_network_root == learned_binding.identity
                and row.binding_witness_roots == learned_binding.witness_roots
                and row.binding_trial_roots == final_selection.binding_trial_roots
                and row.binding_catalyst_witness_roots
                    == final_selection.binding_catalyst_witness_roots
                and row.binding_trajectory_roots
                    == final_selection.binding_trajectory_roots
                and row.binding_trajectory_constituent_roots
                    == final_selection.binding_trajectory_constituent_roots
                and row.binding_surface_apply_roots
                    == final_selection.binding_surface_apply_roots
                and row.nomination_root == final_selection.nomination_root
                and row.ticket_root == final_selection.ticket_root
                and row.ticket_envelope_root == final_selection.ticket_envelope_root
                and row.mapping_evidence_root == final_selection.mapping_evidence_root
                and row.probe_evidence_roots == final_selection.probe_evidence_roots
                and row.apply_receipt_roots == final_selection.apply_receipt_roots
                and row.return_contact_roots == final_selection.return_contact_roots
                and row.outcome_source_roots == final_selection.outcome_source_roots
                and row.leaf.raw_contact_root and row.leaf.sample_root
                and row.leaf.span_occurrence_root and row.leaf.span_recipe_root
                and row.leaf.prediction_witness_root
                for row, unit in zip(final_output.ancestry, final_output.units)))

    before = (catalyst_state(catalyst), surface_state(surface),
              binding_state(binding))
    forged = replace(final_selection, identity=final_selection.identity + 1)
    checks["forged_selection_refuses_atomically"] = (
        refuses(lambda: unfold_resident_partner_surface_v1(
            catalyst, surface, binding, forged), "selection_authentication")
        and before == (catalyst_state(catalyst), surface_state(surface),
                       binding_state(binding)))
    original_command = catalyst.pending.command
    catalyst.pending.command = replace(original_command,
                                       identity=original_command.identity + 1)
    checks["forged_command_refuses_before_voicebox"] = refuses(
        lambda: select_resident_partner_surface_v1(catalyst, surface, binding),
        "command_authentication")
    catalyst.pending.command = original_command
    original_ticket = catalyst.pending.ticket
    catalyst.pending.ticket = replace(original_ticket,
                                      envelope_root=original_ticket.envelope_root + 1)
    checks["stale_or_replaced_live_ticket_refuses"] = refuses(
        lambda: select_resident_partner_surface_v1(catalyst, surface, binding),
        "command_authentication")
    catalyst.pending.ticket = original_ticket
    low_binding = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    low_binding.work_limit = 1; low_binding.work = 0
    low_binding_before = binding_state(low_binding)
    checks["low_binding_budget_refuses_read_only_resolution_atomically"] = (
        refuses(lambda: low_binding.resolve(final_command.action_recipe,
            final_selection.surface_candidate_roots, _BINDING_AUTHORITY),
            "resource")
        and binding_state(low_binding) == low_binding_before)
    low_withdrawal = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    low_withdrawal.work_limit = 1; low_withdrawal.work = 0
    low_withdrawal_before = binding_state(low_withdrawal)
    checks["low_budget_binding_mutation_rolls_back_complete_state"] = (
        refuses(lambda: low_withdrawal.withdraw_source(990001), "resource")
        and binding_state(low_withdrawal) == low_withdrawal_before)
    forged_binding = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    forged_binding.witnesses[0] = replace(
        forged_binding.witnesses[0], catalyst_witness_root=
        forged_binding.witnesses[0].catalyst_witness_root + 1)
    checks["forged_binding_lineage_refuses_before_voicebox"] = refuses(
        lambda: select_resident_partner_surface_v1(
            catalyst, surface, forged_binding), "binding_unresolved")
    low = surface.work_limit; surface.work_limit = 1
    checks["shared_bounded_resource_refuses_before_materialization"] = refuses(
        lambda: select_resident_partner_surface_v1(
            catalyst, surface, binding), "resource")
    surface.work_limit = low
    checks["tamper_and_resource_controls_leave_complete_state_exact"] = (
        before == (catalyst_state(catalyst), surface_state(surface),
                   binding_state(binding)))

    feed(surface, surface_boundary, 3302, CAROL, BOB)
    altered_selection = select_resident_partner_surface_v1(
        catalyst, surface, binding)
    altered_output = unfold_resident_partner_surface_v1(
        catalyst, surface, binding, altered_selection)
    checks["altered_current_occurrence_provenance_diverges_same_surface"] = (
        altered_selection.identity != final_selection.identity
        and altered_output.identity != final_output.identity
        and altered_output.units == learned_units)

    withdrawn_source = final_selection.outcome_source_roots[-1]
    withdrawal = contact_boundary.seal_withdrawal(
        catalyst.session, catalyst.next_sequence, 70001, 79, withdrawn_source)
    catalyst.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_invalidates_pending_consequence_selection"] = (
        catalyst.pending is not None and not catalyst.catalyst_recipes
        and refuses(lambda: select_resident_partner_surface_v1(
            catalyst, surface, binding), "catalyst_relation"))

    wait = catalyst.tick()
    checks["pending_consequence_blocks_next_cognition"] = (
        wait.command is None and catalyst.pending is not None)
    final_evidence = settle(catalyst)
    novel_command = open_episode(catalyst, 60005, novel_span)
    checks["partner_credit_is_context_local_and_unfamiliar_partner_is_silent"] = (
        final_evidence.difference == 1
        and novel_command.action_recipe == ACTION_RECIPES[0].identity
        and refuses(lambda: select_resident_partner_surface_v1(
            catalyst, surface, binding), "catalyst_relation"))
    settle(catalyst)

    feed(surface, surface_boundary, 3303, CAROL, BOB)
    reacquired_binding = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    withdrawn_witness = learned_binding_witnesses[-1]
    withdrawn_apply_root = withdrawn_witness.surface_apply_fields[0]
    reacquired_binding.withdraw_source(withdrawn_apply_root)
    withdrawal_removed_network = not reacquired_binding.networks
    reacquisition_command = open_episode(catalyst, 60007, context_span)
    reacquisition_trial = nominate_resident_action_constructor_trial_v1(
        catalyst, surface, reacquired_binding)
    nomination_materialization_work = (len(reacquisition_trial.candidate_roots)
        + len(reacquisition_trial.source_roots) + 4
        + reacquired_binding._validation_cost())
    combined_nomination_work = reacquired_binding.work
    reacquisition_trajectory = unfold_resident_action_constructor_trial_v1(
        reacquired_binding, surface, reacquisition_trial)
    reacquisition_apply, reacquisition_receipt = (
        bind_resident_action_constructor_surface_v1(
            catalyst, surface, reacquired_binding, response_contract,
            reacquisition_trial, reacquisition_trajectory))
    reacquisition_evidence = settle(catalyst, reacquisition_apply)
    reacquisition_witness = settle_resident_action_constructor_trial_v1(
        catalyst, reacquired_binding)
    reacquired_network = next(iter(reacquired_binding.networks.values()))
    resolved_reacquisition = reacquired_binding.resolve(
        reacquisition_command.action_recipe,
        reacquisition_trial.candidate_roots, _BINDING_AUTHORITY)
    replay_command = open_episode(catalyst, 60008, context_span)
    reacquired_selection = select_resident_partner_surface_v1(
        catalyst, surface, reacquired_binding)
    reacquired_output = unfold_resident_partner_surface_v1(
        catalyst, surface, reacquired_binding, reacquired_selection)
    preflight_binding = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    preflight_binding.withdraw_source(withdrawn_apply_root)
    _command, _ticket, preflight_nomination = _current_command_context(catalyst)
    preflight_candidates, _state, preflight_surface_sources, _work = (
        _surface_candidates(surface))
    preflight_sources = tuple(sorted({*preflight_surface_sources,
                                      *preflight_nomination.source_roots}))
    preflight_binding.work_limit = preflight_binding._nomination_preflight_cost(
        len(preflight_candidates), len(preflight_sources)) - 1
    preflight_before = binding_state(preflight_binding)
    preflight_refused = refuses(lambda:
        nominate_resident_action_constructor_trial_v1(
            catalyst, surface, preflight_binding), "resource")
    checks["ordinary_contact_reacquires_withdrawn_composite_binding"] = (
        withdrawal_removed_network
        and reacquisition_command.action_recipe == learned_binding.action_recipe
        and reacquisition_trial.constructor_root == learned_binding.constructor_root
        and reacquisition_trajectory.units == learned_units
        and reacquisition_evidence.difference == 1
        and body.valid_surface_apply_evidence(reacquisition_receipt)
        and reacquisition_witness.identity in reacquired_network.witness_roots
        and withdrawn_witness.identity not in reacquired_network.witness_roots
        and withdrawn_apply_root in reacquired_binding.withdrawn_sources
        and withdrawn_apply_root not in reacquired_network.source_roots
        and reacquisition_receipt.identity in reacquired_network.source_roots
        and reacquired_network.support == 2 and reacquired_network.credit == 2
        and resolved_reacquisition == reacquired_network)
    checks["nomination_charges_resolution_scan_and_mutation_together"] = (
        combined_nomination_work > nomination_materialization_work)
    checks["nomination_preflight_refuses_one_less_before_mutation"] = (
        preflight_refused
        and binding_state(preflight_binding) == preflight_before)
    checks["reacquired_binding_is_load_bearing_in_composite_byte_trajectory"] = (
        replay_command.action_recipe == learned_binding.action_recipe
        and reacquired_selection.binding_network_root
            == reacquired_network.identity
        and withdrawn_witness.identity
            not in reacquired_selection.binding_witness_roots
        and reacquisition_witness.identity
            in reacquired_selection.binding_witness_roots
        and len(reacquired_output.ancestry) == len(reacquired_output.units)
        and all(row.binding_network_root == reacquired_network.identity
                and withdrawn_witness.identity
                    not in row.binding_witness_roots
                and reacquisition_witness.identity in row.binding_witness_roots
                for row in reacquired_output.ancestry)
        and reacquired_output.units == learned_units)

    withdrawn_binding = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    withdrawn_binding.withdraw_source(
        next(row.outcome_source for row in binding.witnesses
             if row.identity in learned_binding.witness_roots))
    checks["binding_source_withdrawal_cascades_to_silence"] = (
        not withdrawn_binding.networks)
    withdrawn_surface = ResidentActionConstructorBindingV1.restore(
        promoted_binding_checkpoint, binding_boundary)
    withdrawn_surface.withdraw_source(
        learned_binding_witnesses[0].surface_apply_fields[0])
    checks["surface_apply_withdrawal_cascades_to_silence"] = (
        not withdrawn_surface.networks)
    checks["forged_surface_response_contract_refuses"] = (
        not body.valid_surface_contract(replace(
            response_contract, identity=response_contract.identity + 1)))
    corrupt_binding = bytearray(promoted_binding_checkpoint)
    corrupt_binding[-1] ^= 1
    checks["binding_checkpoint_corruption_refuses"] = refuses(
        lambda: ResidentActionConstructorBindingV1.restore(
            corrupt_binding, binding_boundary), "checkpoint")

    checks["body_mapping_contains_only_opaque_physical_commands"] = (
        dict(mapping.entries) == {
            ACTION_RECIPES[0].identity: 9101,
            ACTION_RECIPES[1].identity: 9102}
        and not {9101, 9102}.intersection({regards_root, greets_root}))
    surface_checkpoint = surface.checkpoint()
    catalyst_checkpoint = catalyst.checkpoint()
    binding_checkpoint = binding.checkpoint()
    checks["versioned_complete_component_checkpoints_are_present_and_bounded"] = (
        0 < len(surface_checkpoint) <= (2 << 20)
        and 0 < len(catalyst_checkpoint) <= (8 << 20)
        and 0 < len(binding_checkpoint) <= (2 << 20)
        and b'"schema"' in surface_checkpoint
        and b'"schema"' in catalyst_checkpoint
        and b'"schema"' in binding_checkpoint
        and ResidentActionConstructorBindingV1.restore(
            binding_checkpoint, binding_boundary).checkpoint() == binding_checkpoint)

    elapsed = time.perf_counter() - STARTED
    checks["hard_cpu_runtime_and_state_bounds"] = (
        elapsed < 60 and len(surface._inner.samples) <= 576
        and final_selection.surface_work_units
            + (final_output.work_units - final_selection.work_units)
            <= surface.work_limit
        and final_selection.catalyst_work_units <= catalyst.work_limit
        and binding.work <= binding.work_limit)
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise SystemExit("FOUNDRY_REFERENCE_RESIDENT_PAIR_REAFFERENT_VOICE_RED "
                         + ",".join(failed))
    core = Path(__file__).with_name("reference_resident_partner_surface_v1.py")
    surface_resident_core = Path(__file__).with_name(
        "reference_resident_parametric_span_network_v1.py")
    span_resident_core = Path(__file__).with_name(
        "reference_resident_variable_span_v1.py")
    receipt = {
        "contract": "FOUNDRY_REFERENCE_RESIDENT_PAIR_REAFFERENT_VOICE_BINDING_GREEN",
        "claim": CLAIM, "reference_only": True, "adult_attached": False,
        "runtime_llm": False, "graph_flip": False,
        "fixture_known_expected_bytes": True,
        "host_fixture_current_occurrences": True,
        "host_precommitted_action_constructor_topology": False,
        "host_precommitted_physical_action_codes": True,
        "pair_specific_reafferent_causality": True,
        "external_response_law_authored": True,
        "physical_conversational_causality": False,
        "persistent_state_model": "COMPACT_NUMERIC_RECIPE_RECRUITMENT_MATHEMATICS",
        "transient_state_model": "EPHEMERAL_UNFOLDED_NETWORK_TRAJECTORY",
        "raw_surface_custody": "EXTERNAL_BODY_BOUNDARY_ONLY",
        "ordinary_contact_composite_reacquisition": True,
        "cuda_headers_touched": False, "new_cuda_translation_units": 0,
        "stable_executor_data_driven": True,
        "host_runtime_output_or_partner_selection": False,
        "complete_outputs_seen_in_training": False,
        "human_language_claim": False, "semantic_dialogue_claim": False,
        "trust_or_truth_claim": False, "causal_claim": False,
        "reference_pair_causal_claim": True,
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED", "parity_status": "NOT_RUN/RED",
        "physical_direct_parity": "NOT_RUN/RED",
        "runtime_limit_seconds": 60, "elapsed_ms": round(elapsed * 1000, 3),
        "runtime_headroom_ms": round(60000 - elapsed * 1000, 3),
        "checks": checks,
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "surface_resident_core_sha256": hashlib.sha256(
            surface_resident_core.read_bytes()).hexdigest(),
        "span_resident_core_sha256": hashlib.sha256(
            span_resident_core.read_bytes()).hexdigest(),
        "binding_core_sha256": hashlib.sha256(Path(__file__).with_name(
            "reference_resident_action_constructor_binding_v1.py").read_bytes()).hexdigest(),
        "body_core_sha256": hashlib.sha256(Path(__file__).with_name(
            "reference_resident_authenticated_probe_v1.py").read_bytes()).hexdigest(),
        "active_file_line_counts": {path.name: len(path.read_text().splitlines())
            for path in (core, surface_resident_core, span_resident_core,
                Path(__file__).with_name(
                "reference_resident_action_constructor_binding_v1.py"),
                Path(__file__).with_name(
                    "reference_resident_authenticated_probe_v1.py"),
                Path(__file__))},
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "remaining_red": ["PHYSICAL_CONVERSATIONAL_REAFFERENCE",
            "GROUNDED_PARTNER_IDENTITY", "MULTILINGUAL_RECURSIVE_DIALOGUE",
            "PUBLIC_BODY_CHANNEL", "PRODUCTION_RECIPE_IR_TRANSLATION",
            "DIRECT_PHYSICAL_PARITY", "HUMAN_LANGUAGE_MASTERY"],
    }
    print("FOUNDRY_REFERENCE_RESIDENT_PAIR_REAFFERENT_VOICE_BINDING_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    trained_fixture(assay)
