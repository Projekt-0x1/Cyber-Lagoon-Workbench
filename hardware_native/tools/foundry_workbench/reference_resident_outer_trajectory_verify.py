#!/usr/bin/env python3
"""Hostile assay for a resident-selected transient outer-byte trajectory."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from reference_resident_channel_sequence_grounding_v1 import (
    admit_channel_sequence_boundary_v1,
)
from reference_resident_composite_cue_prediction_v1 import (
    CompositeCueRefuse, ResidentCompositeCuePredictionV1, _identity,
)
from reference_resident_composite_cue_prediction_verify import main as build_chain
from reference_resident_outer_trajectory_v1 import (
    ResidentOuterTrajectoryV1, unfold_resident_outer_trajectory_v1,
)

STARTED = time.perf_counter()


def refuses(call, fragment=""):
    try:
        call()
    except Exception as exc:
        return isinstance(exc, (CompositeCueRefuse, TypeError, ValueError)) and fragment in str(exc)
    return False


def assay(state):
    started = STARTED
    boundary = state["boundary"]
    machine = state["machine"]
    feed_span = state["feed_span"]
    world = state["world"]
    consequence = state["consequence"]
    constructor_l = state["constructor_l"]
    checks = dict(state["checks"])

    feed_span(23101, 8, world)
    phase = machine.tick()
    assert len(phase.prediction_tickets) == 1
    ticket = next(iter(machine.pending.values()))
    before_unfold = machine.checkpoint()
    learning_before = (tuple(machine.nominations), tuple(machine.hypotheses.values()),
                       tuple(machine.recipes.values()), tuple(machine.witnesses),
                       tuple(machine.events), tuple(machine.trace))
    trajectory = unfold_resident_outer_trajectory_v1(machine)
    checks["resident_ticket_selects_learned_outer_bytes"] = (
        isinstance(trajectory, ResidentOuterTrajectoryV1)
        and trajectory.ticket == ticket.ticket
        and trajectory.ticket_envelope_root == ticket.envelope_root
        and trajectory.target_recipe_root == ticket.target_recipe
        and trajectory.units == consequence
        and ticket.prospective_middle_recipe == constructor_l)
    checks["complete_per_byte_causal_ancestry"] = (
        len(trajectory.ancestry) == len(trajectory.units)
        and tuple(row.offset for row in trajectory.ancestry) == tuple(range(len(trajectory.units)))
        and tuple(row.unit for row in trajectory.ancestry) == trajectory.units
        and all(row.raw_contact_root == _identity(b"variable-span-contact-v1", (
                    machine.session, row.contact_sequence, row.source,
                    row.channel, (row.unit,), row.provenance))
                and row.sample_root == _identity(b"variable-span-sample-v1", (
                    row.raw_contact_root, row.contact_sequence))
                and row.source not in machine._inner._inner.withdrawn_sources
                and row.channel == trajectory.target_channel
                and row.target_occurrence_root
                and row.target_recipe_root == trajectory.target_recipe_root
                and row.target_prediction_witness_root in {
                    witness.identity for witness in machine._inner._inner.prediction_witnesses
                    if witness.difference > 0}
                and row.target_prediction_witness_source in row.source_roots
                and row.ticket == trajectory.ticket
                and row.ticket_envelope_root == trajectory.ticket_envelope_root
                and row.cue_node_root == ticket.cue_node
                and row.cue_evidence_revision == ticket.cue_evidence_revision
                and row.relation_recipe_roots == ticket.relation_recipe_roots
                and row.relation_witness_roots == ticket.relation_witness_roots
                and row.relation_source_roots == ticket.relation_source_roots
                and row.source_roots == trajectory.source_roots
                for row in trajectory.ancestry))
    checks["complete_constituent_frontier"] = (
        bool(trajectory.constituent_roots)
        and set(ticket.relation_recipe_roots).issubset(trajectory.constituent_roots)
        and set(ticket.relation_witness_roots).issubset(trajectory.constituent_roots)
        and {row.raw_contact_root for row in trajectory.ancestry}.issubset(
            trajectory.constituent_roots)
        and {row.sample_root for row in trajectory.ancestry}.issubset(
            trajectory.constituent_roots)
        and {row.target_prediction_witness_root for row in trajectory.ancestry}.issubset(
            trajectory.constituent_roots))
    checks["resident_resource_bound_is_load_bearing"] = (
        0 < trajectory.work_units <= machine.work_limit)
    original_limit = machine.work_limit
    machine.work_limit = 1
    low_budget_before = machine.checkpoint()
    checks["low_budget_refuses_before_unfold_without_mutation"] = (
        refuses(lambda: unfold_resident_outer_trajectory_v1(machine), "resource")
        and machine.checkpoint() == low_budget_before)
    machine.work_limit = original_limit
    checks["unfold_is_ephemeral_no_teach_and_atomic"] = (
        machine.checkpoint() == before_unfold
        and learning_before == (tuple(machine.nominations),
            tuple(machine.hypotheses.values()), tuple(machine.recipes.values()),
            tuple(machine.witnesses), tuple(machine.events), tuple(machine.trace)))

    replay = ResidentCompositeCuePredictionV1.restore(before_unfold, boundary)
    replayed = unfold_resident_outer_trajectory_v1(replay)
    checks["same_complete_checkpoint_exact_replay"] = replayed == trajectory

    original = machine.pending[ticket.ticket]
    machine.pending[ticket.ticket] = replace(original, target_recipe=original.target_recipe + 1)
    checks["tampered_ticket_refuses_without_mutation"] = refuses(
        lambda: unfold_resident_outer_trajectory_v1(machine), "ticket")
    machine.pending[ticket.ticket] = original
    machine.pending[ticket.ticket + 999] = replace(original, ticket=ticket.ticket + 999)
    checks["ambiguous_pending_refuses"] = refuses(
        lambda: unfold_resident_outer_trajectory_v1(machine), "pending")
    machine.pending.pop(ticket.ticket + 999)
    checks["literal_or_host_target_bypass_absent"] = (
        len(inspect.signature(unfold_resident_outer_trajectory_v1).parameters) == 1
        and refuses(lambda: unfold_resident_outer_trajectory_v1(machine, consequence)))

    withdrawal_source = ticket.relation_source_roots[0]
    withdrawal = boundary.seal_withdrawal(
        machine.session, machine.next_sequence, 98991, 79, withdrawal_source)
    machine.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_cascades_and_silences"] = (
        not machine.pending
        and refuses(lambda: unfold_resident_outer_trajectory_v1(machine), "pending"))
    checks["altered_authenticated_input_diverges_without_output"] = (
        machine.checkpoint() != before_unfold
        and refuses(lambda: unfold_resident_outer_trajectory_v1(machine), "pending"))

    target_source = trajectory.ancestry[0].source
    target_withdrawal = boundary.seal_withdrawal(
        replay.session, replay.next_sequence, 98992, 79, target_source)
    replay.ingest_withdrawal(target_withdrawal)
    checks["target_byte_source_withdrawal_silences"] = refuses(
        lambda: unfold_resident_outer_trajectory_v1(replay))

    durable = (tuple(machine._inner._inner.recipes.values()), machine.checkpoint())
    forbidden_fields = {"output", "expected", "surface", "semantic", "motor",
                        "route", "prompt", "bytes", "units"}
    recipe_fields = set(next(iter(machine._inner._inner.recipes.values())).__dataclass_fields__)
    checks["durable_recipes_have_no_literal_output_fields"] = (
        not forbidden_fields.intersection(recipe_fields)
        and all(not isinstance(value, (str, bytes, bytearray, memoryview))
                for recipe in durable[0] for value in recipe.__dict__.values()))

    elapsed = time.perf_counter() - started
    checks["hard_runtime_bound"] = elapsed < 60.0
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_REFERENCE_RESIDENT_OUTER_TRAJECTORY_RED " + ",".join(failed))

    here = Path(__file__).parent
    paths = [here / "reference_resident_outer_trajectory_v1.py",
             here / "reference_resident_outer_trajectory_verify.py",
             here / "reference_resident_composite_cue_prediction_verify.py"]
    receipt = {
        "contract": "FOUNDRY_REFERENCE_RESIDENT_OUTER_TRAJECTORY_GREEN",
        "claim": "RESIDENT_SELECTED_LEARNED_OUTER_BYTE_TRAJECTORY_REFERENCE_ONLY",
        "reference_only": True, "adult_attached": False, "runtime_llm": False,
        "host_numeric_fixture": True, "host_target_selection": False,
        "fixture_known_expected_bytes": True,
        "surface_bytes_in_recipe": False, "trajectory_feedback": False,
        "human_language_claim": False, "semantic_claim": False,
        "causal_claim": False, "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED", "parity_status": "NOT_RUN/RED",
        "executor": "CPU_ONLY_STABLE_PYTHON_REFERENCE",
        "runtime_limit_seconds": 60, "elapsed_ms": round(elapsed * 1000, 3),
        "checks": checks,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths},
        "remaining_red": ["NATURAL_LANGUAGE_ACQUISITION", "SEMANTIC_WORLD_REFERENCE",
                          "PHYSICAL_COMMON_CAUSE", "PUBLIC_BODY_CHANNEL",
                          "PRODUCTION_RECIPE_IR_TRANSLATION", "DIRECT_PHYSICAL_PARITY"],
    }
    print("FOUNDRY_REFERENCE_RESIDENT_OUTER_TRAJECTORY_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt


if __name__ == "__main__":
    build_chain(fixture_hook=assay)
