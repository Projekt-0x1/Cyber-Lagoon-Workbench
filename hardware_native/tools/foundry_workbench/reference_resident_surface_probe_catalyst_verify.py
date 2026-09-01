#!/usr/bin/env python3
"""Hostile CPU assay for a resident learned-surface probe catalyst."""
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
import reference_resident_relation_ir_v1 as relation_ir
import reference_resident_variable_span_v1 as span_module
from reference_resident_authenticated_probe_v1 import (
    ACTION_RECIPES, AuthenticatedProbeRefuse, PROBE_HORIZON,
    admit_body_probe_boundary_v1,
)
from reference_resident_composite_cue_prediction_verify import main as trained_fixture
from reference_resident_surface_probe_catalyst_v1 import (
    ResidentSurfaceProbeCatalystV1, SurfaceProbeCatalystRecipeV1,
)


CLAIM = "RESIDENT_LEARNED_SURFACE_PROBE_CATALYST_REFERENCE_ONLY"


def refuses(call, fragment=""):
    try:
        call()
    except Exception as exc:
        return isinstance(exc, (RuntimeError, TypeError, ValueError)) and fragment in str(exc)
    return False


class ReversedEnumerationCatalystV1(ResidentSurfaceProbeCatalystV1):
    def _eligible(self, ticket):
        return tuple(reversed(super()._eligible(ticket)))


class FaultAfterPromotionCatalystV1(ResidentSurfaceProbeCatalystV1):
    def _rebuild_catalysts(self):
        super()._rebuild_catalysts()
        if self.catalyst_recipes:
            raise AuthenticatedProbeRefuse("surface_probe:injected_post_rebuild")


def assay(ctx):
    started = time.perf_counter()
    boundary, trained = ctx["boundary"], ctx["machine"]
    world, consequence, distractor = ctx["world"], ctx["consequence"], ctx["distractor"]
    learned = ctx["learned"]
    checks = dict(ctx["checks"])
    outer = sorted(span for (channel, span) in learned if channel == 7 and span[0] < 200)
    context_span, novel_span = outer[:2]
    seed = probe._pack_inner(trained)

    def make_body(device, body_epoch, commands, negative_sources, positive_sources):
        body = admit_body_probe_boundary_v1(boundary)
        routes = (
            (ACTION_RECIPES[0].identity, commands[0], negative_sources, 9, distractor),
            (ACTION_RECIPES[1].identity, commands[1], positive_sources, 9, consequence),
        )
        mapping = body.precommit(trained.session, device, body_epoch, 3, routes)
        return body, mapping

    body, mapping = make_body(91, 31, (7101, 7102),
                              (41001, 41002, 41003, 41004),
                              (42001, 42002, 42003, 42004))
    route_state = body._snapshot_state()
    checks["source_schedule_is_precommitted_bounded_and_host_invisible"] = (
        "source" not in inspect.signature(body.seal_return).parameters
        and refuses(lambda: body.precommit(trained.session, 93, 33, 3, (
            (ACTION_RECIPES[0].identity, 1, (), 9, distractor),
            (ACTION_RECIPES[1].identity, 2, (1, 1), 9, consequence))),
            "mapping_sources")
        and body._snapshot_state() == route_state)
    resident = ResidentSurfaceProbeCatalystV1(
        boundary, body, mapping,
        initial_inner=probe._unpack_inner(seed, boundary))

    def put(machine, source, channel, unit):
        contact = boundary.seal_sample(
            machine.session, machine.next_sequence, source, channel, (unit,),
            (source, channel))
        return machine.ingest_sample(contact)

    def open_episode(machine, source, span=context_span):
        if span is not None:
            for unit in span:
                put(machine, source, 7, unit)
        for unit in world:
            put(machine, source, 8, unit)
        result = machine.tick()
        assert result.command is not None
        return result.command

    def settle(machine, active_body):
        active_mapping = machine.mapping
        apply = machine.dispatch()
        returns, evidence = [], None
        while machine.pending is not None:
            contact = active_body.seal_return(
                active_mapping, apply, machine.next_sequence)
            returns.append(contact)
            evidence = machine.ingest_return(contact)
        assert evidence is not None
        return apply, tuple(returns), evidence

    commands, episodes, prepromotion = [], [], None
    for index, source in enumerate((50001, 50002, 50003)):
        command = open_episode(resident, source)
        apply, returns, evidence = settle(resident, body)
        commands.append(command); episodes.append((apply, returns, evidence))
        if index == 1:
            prepromotion = resident.checkpoint()

    recipe_rows = tuple(resident.catalyst_recipes.values())
    witness_sources = tuple(row.outcome_source for row in resident.catalyst_witnesses)
    checks["resident_credit_transition_a_minus_b_plus_b_plus"] = (
        tuple(row.action_recipe for row in commands)
        == (ACTION_RECIPES[0].identity, ACTION_RECIPES[1].identity,
            ACTION_RECIPES[1].identity)
        and tuple(row[2].difference for row in episodes) == (-1, 1, 1))
    checks["promotion_requires_two_authenticated_positive_sources"] = (
        len(recipe_rows) == 1 and len(set(witness_sources[1:])) == 2
        and witness_sources[1:] == (42001, 42002)
        and recipe_rows[0].credit == 2)
    checks["atomic_condensation_consumes_global_probe_evidence"] = (
        not resident.evidence
        and set(recipe_rows[0].consumed_probe_evidence_roots)
        == {row[2].identity for row in episodes}
        and len(resident.catalyst_witnesses) == 3)
    checks["compact_recipe_has_no_surface_or_action_payload"] = (
        tuple(row.name for row in fields(SurfaceProbeCatalystRecipeV1))
        == ("identity", "span_recipe", "mapping_root", "credit",
            "witness_roots", "consumed_probe_evidence_roots", "source_roots")
        and not any(hasattr(recipe_rows[0], name) for name in
                    ("action", "action_recipe", "units", "bytes", "output",
                     "expected", "semantic", "motor", "route")))
    ir_program = relation_ir.unfold_bound_relation_v1(
        recipe_rows[0].span_recipe, recipe_rows[0].mapping_root,
        recipe_rows[0].credit, recipe_rows[0].witness_roots,
        recipe_rows[0].source_roots)
    ir_roots = set(recipe_rows[0].witness_roots)
    ir_bindings = tuple(relation_ir.ResidentRelationBindingV1(
        row.identity, row.action_recipe, row.source_roots)
        for row in resident.catalyst_witnesses if row.identity in ir_roots)
    def ir_value(span_recipe, mapping_root, action_recipe):
        return relation_ir.execute_relation_ir_v1(
            ir_program, relation_ir.ResidentRelationFrameV1(
                (span_recipe, mapping_root, action_recipe), ir_bindings))
    ir_positive = ir_value(
        recipe_rows[0].span_recipe, mapping.identity, ACTION_RECIPES[1].identity)
    checks["compact_recipe_unfolds_bounded_typed_relation_ir"] = (
        ir_program.version == relation_ir.IR_VERSION
        and len(ir_program.instructions) <= relation_ir.MAX_INSTRUCTIONS
        and ir_positive.value == 4 * recipe_rows[0].credit
        and ir_positive.binding_roots == recipe_rows[0].witness_roots
        and ir_positive.work <= relation_ir.MAX_EXECUTION_WORK
        and not any(row.name in {"bytes", "output", "expected", "semantic",
                                 "motor", "route", "action"}
                    for row in fields(relation_ir.ResidentRelationIrProgramV1)))
    checks["relation_ir_requires_all_current_bindings_without_semantic_dispatch"] = (
        ir_value(recipe_rows[0].span_recipe, mapping.identity,
                 ACTION_RECIPES[0].identity).value == 0
        and ir_value(learned[(7, novel_span)], mapping.identity,
                     ACTION_RECIPES[1].identity).value == 0
        and ir_value(recipe_rows[0].span_recipe, mapping.identity + 1,
                     ACTION_RECIPES[1].identity).value == 0)
    literal_ir = replace(ir_program, constants=ir_program.constants + ("surface",))
    missing_binding_frame = relation_ir.ResidentRelationFrameV1(
        (recipe_rows[0].span_recipe, mapping.identity, ACTION_RECIPES[1].identity),
        ir_bindings[:-1])
    checks["malformed_literal_and_missing_binding_ir_refuse"] = (
        refuses(lambda: relation_ir.validate_program_v1(literal_ir), "relation_ir:type")
        and refuses(lambda: relation_ir.execute_relation_ir_v1(
            ir_program, missing_binding_frame), "relation_ir:binding"))
    unconditional_shell = replace(
        ir_program, identity=0, instructions=(
            relation_ir.ResidentRelationInstructionV1(
                relation_ir.OP_ACCUMULATE_PRODUCT, 2, 3),
            relation_ir.ResidentRelationInstructionV1(relation_ir.OP_HALT, 0, 0)),
        witness_roots=(), source_roots=())
    unconditional = replace(unconditional_shell, identity=relation_ir._identity(
        b"resident-relation-ir-program-v1",
        relation_ir._program_values(unconditional_shell)))
    malformed_instruction = replace(ir_program, instructions=((1, 0, 0),))
    empty_ancestry = replace(ir_bindings[0], source_roots=())
    empty_ancestry_frame = relation_ir.ResidentRelationFrameV1(
        (recipe_rows[0].span_recipe, mapping.identity, ACTION_RECIPES[1].identity),
        (empty_ancestry,) + ir_bindings[1:])
    checks["unguarded_program_malformed_instruction_and_empty_ancestry_refuse"] = (
        refuses(lambda: relation_ir.validate_program_v1(unconditional),
                "relation_ir:program")
        and refuses(lambda: relation_ir.validate_program_v1(malformed_instruction),
                    "relation_ir:instruction_type")
        and refuses(lambda: relation_ir.execute_relation_ir_v1(
            ir_program, empty_ancestry_frame), "relation_ir:binding"))
    ancestry_ok = True
    nominations = {row.identity: row for row in resident.context_nominations}
    for catalyst, (apply, returns, evidence) in zip(
            resident.catalyst_witnesses, episodes):
        structural = next(row for row in resident._inner.witnesses
                          if row.identity == evidence.structural_witness)
        raw_roots = tuple(span_module._identity(
            b"variable-span-contact-v1", contact.occurrence.signed_fields())
            for contact in returns)
        ancestry_ok &= (
            len(returns) == len(structural.ancestry) == len(consequence)
            and tuple(row.raw_contact_root for row in structural.ancestry) == raw_roots
            and tuple(row.offset for row in structural.ancestry) == tuple(range(len(returns)))
            and tuple(row.unit for row in structural.ancestry)
            == tuple(contact.occurrence.features[0] for contact in returns)
            and all(contact.occurrence.source == catalyst.outcome_source
                    and contact.occurrence.channel == 9
                    and contact.device == mapping.device
                    and contact.body_epoch == mapping.body_epoch
                    and contact.apply_receipt == apply.identity
                    and contact.occurrence.provenance[-1] == apply.identity
                    for contact in returns)
            and evidence.return_roots == tuple(contact.identity for contact in returns)
            and catalyst.nomination in nominations
            and nominations[catalyst.nomination].span_recipe == catalyst.span_recipe
            and set(evidence.source_roots).issubset(set(catalyst.source_roots)))
    checks["per_byte_return_and_nomination_ancestry_reaches_catalyst"] = ancestry_ok

    promoted = resident.checkpoint()
    checkpoint_keys = set()
    def collect_keys(value):
        if isinstance(value, dict):
            checkpoint_keys.update(value)
            for item in value.values(): collect_keys(item)
        elif isinstance(value, list):
            for item in value: collect_keys(item)
    collect_keys(json.loads(promoted))
    checks["unfolded_ir_is_ephemeral_not_checkpoint_state"] = not checkpoint_keys.intersection(
        {"instructions", "constants", "input_arity"})
    replay = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    replay_blob = replay.checkpoint()
    checks["versioned_complete_checkpoint_exact"] = replay_blob == promoted
    body_before_corrupt = body._snapshot_state()
    corrupt = promoted[:-1] + bytes((promoted[-1] ^ 1,))
    checks["corrupt_checkpoint_refuses_without_body_mutation"] = (
        refuses(lambda: ResidentSurfaceProbeCatalystV1.restore(
            corrupt, boundary, body), "checkpoint")
        and body._snapshot_state() == body_before_corrupt)
    literal = replay
    literal_recipe = next(iter(literal.catalyst_recipes.values()))
    def install_literal_recipe():
        literal.catalyst_recipes[literal_recipe.identity] = replace(
            literal_recipe, source_roots=literal_recipe.source_roots + ("surface",))
    literal_body_before = body._snapshot_state()
    checks["literal_state_bypass_refuses_before_checkpoint_signing"] = (
        refuses(install_literal_recipe, "does not support item assignment")
        and literal.checkpoint() == replay_blob
        and body._snapshot_state() == literal_body_before)

    contextual = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    contextual_command = open_episode(contextual, 51001)
    omitted = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    omitted_command = open_episode(omitted, 51002, None)
    novel = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    novel_command = open_episode(novel, 51003, novel_span)
    checks["current_learned_context_rederives_alternate_action"] = (
        contextual_command.action_recipe == ACTION_RECIPES[1].identity)
    checks["omitted_and_unpromoted_context_use_authored_baseline"] = (
        omitted_command.action_recipe == novel_command.action_recipe
        == ACTION_RECIPES[0].identity)
    checks["altered_contact_diverges_without_host_selection"] = (
        len({contextual_command.identity, omitted_command.identity,
             novel_command.identity}) == 3)

    # A wrong authenticated-body field cannot partially advance either resident or body.
    wrong = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    open_episode(wrong, 52001); wrong_apply = wrong.dispatch()
    proper = body.seal_return(mapping, wrong_apply, wrong.next_sequence)
    wrong_before = wrong.checkpoint()
    forged = replace(proper, device=proper.device + 1)
    checks["wrong_consequence_refuses_atomically"] = (
        refuses(lambda: wrong.ingest_return(forged), "return_authentication")
        and wrong.checkpoint() == wrong_before)

    # Exact replay also covers a partially arrived multi-byte consequence.
    mid = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    open_episode(mid, 53001); mid_apply = mid.dispatch()
    first_return = body.seal_return(mapping, mid_apply, mid.next_sequence)
    assert mid.ingest_return(first_return) is None
    mid_blob = mid.checkpoint()

    def finish_from(blob):
        machine = ResidentSurfaceProbeCatalystV1.restore(blob, boundary, body)
        apply = machine.pending.apply_receipt
        roots, evidence = [], None
        while machine.pending is not None:
            contact = body.seal_return(mapping, apply, machine.next_sequence)
            roots.append(contact.identity); evidence = machine.ingest_return(contact)
        return tuple(roots), evidence.identity, machine.checkpoint()

    checks["mid_consequence_checkpoint_exact_replay"] = (
        finish_from(mid_blob) == finish_from(mid_blob))

    # Failure after witness append and recipe materialization must undo the whole resident step.
    fault = FaultAfterPromotionCatalystV1.restore(
        prepromotion, boundary, body)
    open_episode(fault, 53501); fault_apply = fault.dispatch()
    for _ in range(len(consequence) - 1):
        contact = body.seal_return(mapping, fault_apply, fault.next_sequence)
        assert fault.ingest_return(contact) is None
    final_contact = body.seal_return(mapping, fault_apply, fault.next_sequence)
    fault_before = fault.checkpoint()
    checks["post_rebuild_failure_rolls_back_complete_resident_transaction"] = (
        refuses(lambda: fault.ingest_return(final_contact), "injected_post_rebuild")
        and fault.checkpoint() == fault_before
        and len(fault.catalyst_witnesses) == 2
        and not fault.catalyst_recipes)

    # Withholding a consequence cannot teach or leave a pending cognitive episode.
    no_teach = ResidentSurfaceProbeCatalystV1(
        boundary, body, mapping,
        initial_inner=probe._unpack_inner(seed, boundary))
    open_episode(no_teach, 54001); no_teach.dispatch()
    for _ in range(PROBE_HORIZON):
        no_teach.tick()
    checks["withheld_consequence_no_teach"] = (
        no_teach.pending is None and not no_teach.evidence
        and not no_teach.catalyst_witnesses and not no_teach.catalyst_recipes)

    # Removing one causal outcome collapses the recipe; new lived evidence can rebuild it.
    withdrawn = ResidentSurfaceProbeCatalystV1.restore(promoted, boundary, body)
    withdrawal = boundary.seal_withdrawal(
        withdrawn.session, withdrawn.next_sequence, 70001, 79, 42001)
    withdrawn.ingest_withdrawal(withdrawal)
    collapsed = not withdrawn.catalyst_recipes
    reacquired_actions = []
    for source in (55001, 55002, 55003):
        command = open_episode(withdrawn, source)
        reacquired_actions.append(command.action_recipe)
        settle(withdrawn, body)
    checks["withdrawal_cascades_and_new_contact_reacquires"] = (
        collapsed and tuple(reacquired_actions)
        == (ACTION_RECIPES[0].identity, ACTION_RECIPES[1].identity,
            ACTION_RECIPES[1].identity)
        and len(withdrawn.catalyst_recipes) == 1)

    # The compact edge is the unfolded program: skipping it lesions the delta;
    # an independently rematerialized program is byte-identical and restores it.
    rematerialized_program = relation_ir.unfold_bound_relation_v1(
        recipe_rows[0].span_recipe, recipe_rows[0].mapping_root,
        recipe_rows[0].credit, recipe_rows[0].witness_roots,
        recipe_rows[0].source_roots)
    rematerialized_value = relation_ir.execute_relation_ir_v1(
        rematerialized_program, relation_ir.ResidentRelationFrameV1(
            (recipe_rows[0].span_recipe, mapping.identity,
             ACTION_RECIPES[1].identity), ir_bindings)).value
    checks["compact_lesion_sham_and_rematerialization"] = (
        0 != ir_positive.value == rematerialized_value
        and rematerialized_program == ir_program)

    # Repeated authenticated returns from one provenance label never satisfy corroboration.
    yoked_body, yoked_mapping = make_body(
        94, 34, (8101, 8102), (45001,), (46001,))
    yoked = ResidentSurfaceProbeCatalystV1(
        boundary, yoked_body, yoked_mapping,
        initial_inner=probe._unpack_inner(seed, boundary))
    yoked_actions = []
    for source in (57001, 57002, 57003):
        command = open_episode(yoked, source)
        yoked_actions.append(command.action_recipe); settle(yoked, yoked_body)
    checks["repeated_same_source_positive_returns_do_not_promote"] = (
        tuple(yoked_actions)
        == (ACTION_RECIPES[0].identity, ACTION_RECIPES[1].identity,
            ACTION_RECIPES[1].identity)
        and not yoked.catalyst_recipes)

    # Enumeration order and opaque command slots cannot select the action.
    body2, mapping2 = make_body(92, 32, (9902, 9901),
                               (43001, 43002, 43003),
                               (44001, 44002, 44003))
    permuted = ReversedEnumerationCatalystV1(
        boundary, body2, mapping2,
        initial_inner=probe._unpack_inner(seed, boundary))
    permuted_actions = []
    for source in (60001, 60002, 60003):
        command = open_episode(permuted, source)
        permuted_actions.append(command.action_recipe); settle(permuted, body2)
    permuted_command = open_episode(permuted, 60004)
    checks["opaque_slot_and_enumeration_permutation_invariant"] = (
        tuple(permuted_actions)
        == (ACTION_RECIPES[0].identity, ACTION_RECIPES[1].identity,
            ACTION_RECIPES[1].identity)
        and permuted_command.action_recipe == ACTION_RECIPES[1].identity)

    constructor_parameters = set(inspect.signature(
        ResidentSurfaceProbeCatalystV1).parameters)
    checks["no_host_current_thought_or_literal_bypass_api"] = (
        not constructor_parameters.intersection({
            "prompt", "context", "candidate", "action", "output", "expected",
            "semantic", "motor", "route"})
        and not hasattr(resident, "enqueue_goal")
        and not hasattr(resident, "install_catalyst"))
    checks["bounded_cpu_reference_runtime"] = (
        time.perf_counter() - started < 60
        and resident.work <= resident.work_limit
        and len(resident.catalyst_witnesses) <= 512
        and len(resident.catalyst_recipes) <= 128)

    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    payload = {
        "claim": CLAIM,
        "reference_only": True,
        "adult_status": "RED",
        "language_status": "RED",
        "semantic_status": "RED",
        "physical_causal_status": "RED",
        "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "experimental_ir": "ResidentRelationIrProgramV1",
        "production_translation_owner": "g.reflective_recipe_kernel_closure",
        "direct_genome_role": "BIRTH_ONLY_NOT_LEARNED_RELATION_TARGET",
        "translation_status": "UNDEFINED",
        "parity_status": "NOT_RUN",
        "graph_flip": False,
        "runtime_llm": False,
        "host_current_thought": False,
        "executor": "CPython graph-neutral deterministic reference",
        "check_count": len(checks),
        "checks": checks,
        "mapping_root": mapping.identity,
        "catalyst_recipe_root": recipe_rows[0].identity,
        "catalyst_witness_roots": list(recipe_rows[0].witness_roots),
        "positive_outcome_sources": list(witness_sources[1:]),
        "source_authority_status": "AUTHENTICATED_REFERENCE_LABELS_NOT_PHYSICAL_INDEPENDENCE",
        "checkpoint_sha256": hashlib.sha256(promoted).hexdigest(),
        "checkpoint_byte_bound": probe.MAX_CHECKPOINT_BYTES,
        "resident_call_work_limit": resident.work_limit,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    print("FOUNDRY_REFERENCE_RESIDENT_SURFACE_PROBE_CATALYST_GREEN")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


if __name__ == "__main__":
    trained_fixture(assay)
